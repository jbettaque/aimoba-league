/**
 * A small zip reader and writer (no dependencies; `node:zlib` does the compression). The league reads entries out
 * of the archives GitHub makes of a commit.
 *
 * The reader is written for archives from strangers: it refuses anything that could write outside the target
 * directory (absolute paths, `..`, drive letters), symbolic links, encrypted or zip64 entries, and it never
 * inflates more than the limits allow, whatever sizes the archive claims.
 */
import { crc32, deflateRawSync, inflateRawSync } from 'node:zlib';

export interface ZipEntry { name: string; data: Buffer }
export interface ZipLimits {
  /** Most files taken out of the archive. */
  maxFiles: number;
  maxTotalBytes: number;
  maxFileBytes: number;
  /** Most entries the archive may list at all, the skipped ones included (default: twice maxFiles). */
  maxEntries?: number;
}

/**
 * Picks the entries to take out and renames them: return the name to give the file, or null to skip it. A skipped
 * entry is never unpacked and does not count toward the limits (other than `maxEntries`), whatever it is.
 */
export type ZipSelect = (rawName: string) => string | null;

export const DEFAULT_ZIP_LIMITS: ZipLimits = { maxFiles: 500, maxTotalBytes: 64 * 1024 * 1024, maxFileBytes: 48 * 1024 * 1024 };

export class ZipError extends Error {}

/** A safe relative path with forward slashes, or null if the name must not be extracted. */
export function safeName(raw: string): string | null {
  const name = raw.split('\\').join('/');
  if (!name || name.includes('\0') || name.startsWith('/') || /^[a-zA-Z]:/.test(name)) return null;
  const parts = name.split('/');
  if (parts.some((p) => p === '..' || p === '.' || p === '')) return null;
  if (name.length > 240 || parts.some((p) => p.length > 120)) return null;
  // Names Windows cannot create, or that would land on a device.
  if (parts.some((p) => /[<>:"|?*]/.test(p) || /^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$/i.test(p) || p.endsWith('.') || p.endsWith(' '))) return null;
  return name;
}

export function readZip(buf: Buffer, limits: ZipLimits = DEFAULT_ZIP_LIMITS, select?: ZipSelect): ZipEntry[] {
  // The end-of-central-directory record is within the last 64 KiB + 22 bytes.
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 22 - 0xffff); i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new ZipError('not a zip archive');
  const count = buf.readUInt16LE(eocd + 10);
  const dirSize = buf.readUInt32LE(eocd + 12), dirOffset = buf.readUInt32LE(eocd + 16);
  if (count === 0xffff || dirOffset === 0xffffffff || dirSize === 0xffffffff) throw new ZipError('zip64 archives are not supported');
  if (dirOffset + dirSize > eocd) throw new ZipError('the archive is damaged (central directory out of bounds)');
  const maxEntries = limits.maxEntries ?? limits.maxFiles * 2;
  if (count > maxEntries) throw new ZipError(`too many entries in the archive (${count}; limit ${maxEntries})`);

  const out: ZipEntry[] = [];
  const seen = new Set<string>();
  let total = 0, p = dirOffset;
  for (let i = 0; i < count; i++) {
    if (p + 46 > buf.length || buf.readUInt32LE(p) !== 0x02014b50) throw new ZipError('the archive is damaged (bad central directory entry)');
    const madeBy = buf.readUInt16LE(p + 4), flags = buf.readUInt16LE(p + 8), method = buf.readUInt16LE(p + 10);
    const crc = buf.readUInt32LE(p + 16), csize = buf.readUInt32LE(p + 20), usize = buf.readUInt32LE(p + 24);
    const nlen = buf.readUInt16LE(p + 28), xlen = buf.readUInt16LE(p + 30), clen = buf.readUInt16LE(p + 32);
    const attrs = buf.readUInt32LE(p + 38), local = buf.readUInt32LE(p + 42);
    const rawName = buf.subarray(p + 46, p + 46 + nlen).toString('utf8');
    p += 46 + nlen + xlen + clen;

    if (rawName.endsWith('/') || rawName.endsWith('\\')) continue; // a directory
    let wanted = rawName;
    if (select) {
      const picked = select(rawName);
      if (picked === null) continue;
      wanted = picked;
    }
    const unixMode = madeBy >> 8 === 3 ? attrs >>> 16 : 0;
    if ((unixMode & 0o170000) === 0o120000) throw new ZipError(`${rawName}: symbolic links are not allowed`);
    if (unixMode && (unixMode & 0o170000) !== 0o100000 && (unixMode & 0o170000) !== 0) throw new ZipError(`${rawName}: only regular files are allowed`);
    if (flags & 1) throw new ZipError(`${rawName}: encrypted entries are not supported`);
    if (csize === 0xffffffff || usize === 0xffffffff || local === 0xffffffff) throw new ZipError('zip64 archives are not supported');
    if (method !== 0 && method !== 8) throw new ZipError(`${rawName}: unsupported compression method ${method}`);
    const name = safeName(wanted);
    if (!name) throw new ZipError(`unsafe or invalid file name in the archive: ${JSON.stringify(wanted.slice(0, 80))}`);
    if (seen.has(name.toLowerCase())) throw new ZipError(`${name}: the same file appears twice`);
    seen.add(name.toLowerCase());
    if (out.length + 1 > limits.maxFiles) throw new ZipError(`too many files (limit ${limits.maxFiles})`);
    if (usize > limits.maxFileBytes) throw new ZipError(`${name} is too large (${usize} bytes; limit ${limits.maxFileBytes})`);
    total += usize;
    if (total > limits.maxTotalBytes) throw new ZipError(`the archive unpacks to more than ${limits.maxTotalBytes} bytes`);

    if (local + 30 > buf.length || buf.readUInt32LE(local) !== 0x04034b50) throw new ZipError(`${name}: the archive is damaged (bad local header)`);
    const start = local + 30 + buf.readUInt16LE(local + 26) + buf.readUInt16LE(local + 28);
    if (start + csize > buf.length) throw new ZipError(`${name}: the archive is damaged (data out of bounds)`);
    const raw = buf.subarray(start, start + csize);
    let data: Buffer;
    if (method === 0) data = Buffer.from(raw);
    else {
      try {
        // Never trust the claimed size: stop inflating one byte past it.
        data = inflateRawSync(raw, { maxOutputLength: usize + 1 });
      } catch (err) {
        throw new ZipError(`${name}: could not be unpacked (${(err as Error).message})`);
      }
    }
    if (data.length !== usize) throw new ZipError(`${name}: the archive is damaged (size mismatch)`);
    if ((crc32(data) >>> 0) !== crc) throw new ZipError(`${name}: the archive is damaged (checksum mismatch)`);
    out.push({ name, data });
  }
  return out;
}

/** Pack files into a zip archive. Names use forward slashes. The result is deterministic for the same input. */
export function writeZip(entries: ZipEntry[]): Buffer {
  const parts: Buffer[] = [];
  const central: Buffer[] = [];
  let offset = 0;
  for (const e of entries) {
    const name = Buffer.from(e.name, 'utf8');
    const packed = deflateRawSync(e.data, { level: 9 });
    const store = packed.length >= e.data.length;
    const body = store ? e.data : packed;
    const crc = crc32(e.data) >>> 0;
    const head = Buffer.alloc(30);
    head.writeUInt32LE(0x04034b50, 0);
    head.writeUInt16LE(20, 4);
    head.writeUInt16LE(0x0800, 6); // UTF-8 names
    head.writeUInt16LE(store ? 0 : 8, 8);
    head.writeUInt16LE(0, 10); // a fixed time and date (1980-01-01), so equal inputs give equal archives
    head.writeUInt16LE(0x21, 12);
    head.writeUInt32LE(crc, 14);
    head.writeUInt32LE(body.length, 18);
    head.writeUInt32LE(e.data.length, 22);
    head.writeUInt16LE(name.length, 26);
    parts.push(head, name, body);
    const dir = Buffer.alloc(46);
    dir.writeUInt32LE(0x02014b50, 0);
    dir.writeUInt16LE(20, 4);
    dir.writeUInt16LE(20, 6);
    dir.writeUInt16LE(0x0800, 8);
    dir.writeUInt16LE(store ? 0 : 8, 10);
    dir.writeUInt16LE(0, 12);
    dir.writeUInt16LE(0x21, 14);
    dir.writeUInt32LE(crc, 16);
    dir.writeUInt32LE(body.length, 20);
    dir.writeUInt32LE(e.data.length, 24);
    dir.writeUInt16LE(name.length, 28);
    dir.writeUInt32LE(offset, 42);
    central.push(dir, name);
    offset += 30 + name.length + body.length;
  }
  const dirSize = central.reduce((n, b) => n + b.length, 0);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(dirSize, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...parts, ...central, end]);
}
