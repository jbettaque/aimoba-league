/**
 * Where an entry's code lives: a folder in a public GitHub repository. Shared by the league server (which fetches
 * entries from there) and the command line (which works out the repository, folder and commit of a local bot).
 */

export interface RepoRef { owner: string; repo: string }

const NAME = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,99})$/;

/**
 * Reads a repository from what people type: https://github.com/owner/repo(.git)(/tree/<ref>/<path>), github.com/owner/repo,
 * git@github.com:owner/repo.git or owner/repo. A /tree/ or /blob/ link also gives a ref and a path. Null for anything
 * that is not a GitHub repository.
 */
export function parseRepo(input: string): (RepoRef & { ref?: string; path?: string }) | null {
  let s = String(input ?? '').trim();
  if (!s) return null;
  s = s.replace(/^git@github\.com:/i, 'https://github.com/').replace(/^(?:git\+)?ssh:\/\/git@github\.com\//i, 'https://github.com/');
  if (/^(www\.)?github\.com\//i.test(s)) s = `https://${s}`;
  let parts: string[];
  if (/^[A-Za-z0-9][\w.-]*\/[\w.-]+$/.test(s)) parts = s.split('/');
  else {
    let url: URL;
    try { url = new URL(s); } catch { return null; }
    if (!/^https?:$/.test(url.protocol) || !/^(www\.)?github\.com$/i.test(url.hostname) || url.username || url.password) return null;
    try { parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent); } catch { return null; }
  }
  if (parts.length < 2) return null;
  const owner = parts[0], repo = parts[1].replace(/\.git$/i, '');
  if (!NAME.test(owner) || !NAME.test(repo) || repo === '.' || repo === '..') return null;
  const out: RepoRef & { ref?: string; path?: string } = { owner, repo };
  // https://github.com/o/r/tree/<ref>/<path>: the ref is taken to be one segment (a branch with a slash in it must be given separately).
  if ((parts[2] === 'tree' || parts[2] === 'blob') && parts[3]) {
    out.ref = parts[3];
    const rest = parts.slice(4);
    if (parts[2] === 'blob' && rest.length) rest.pop(); // a link to a file means its folder
    if (rest.length) out.path = rest.join('/');
  }
  return out;
}

/** A folder in a repository, normalised: forward slashes, no leading or trailing slash. '' is the top level; null if it is not a valid folder. */
export function normalizePath(path: string | undefined): string | null {
  const p = String(path ?? '').trim().split('\\').join('/').replace(/^\.?\/+/, '').replace(/\/+$/, '');
  if (!p || p === '.') return '';
  const parts = p.split('/');
  if (parts.some((x) => !x || x === '.' || x === '..' || x.length > 120) || p.length > 240) return null;
  return parts.join('/');
}

export const isCommitId = (ref: string) => /^[0-9a-f]{40}$/i.test(ref);
export const repoUrl = (r: RepoRef) => `https://github.com/${r.owner}/${r.repo}`;
/** The page of the entry's folder at a commit (or a branch). */
export const sourceUrl = (r: RepoRef, commit: string, path: string) =>
  `${repoUrl(r)}/tree/${encodeURIComponent(commit)}${path ? `/${path.split('/').map(encodeURIComponent).join('/')}` : ''}`;
