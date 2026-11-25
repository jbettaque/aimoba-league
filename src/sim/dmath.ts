/**
 * Math for the simulation that gives the same bits on every JavaScript engine.
 *
 * `Math.sin`, `cos`, `atan2`, `hypot` and `**` are only specified approximately, and Node, Chrome and Firefox
 * do differ in the last digit. One such digit is enough for a replay recorded in one engine to drift apart
 * in another. Addition, subtraction, multiplication, division and `Math.sqrt` are exact by the language
 * standard, so everything here is built from those alone (kernels and constants after FreeBSD's msun / fdlibm).
 *
 * The simulation (src/sim, the scripted bots included) must use these instead of the `Math` versions.
 */

const INV_PIO2 = 6.36619772367581382433e-1;
/** pi/2 split in two, so that `x - k * pi/2` keeps its low bits. */
const PIO2_HI = 1.57079632673412561417, PIO2_LO = 6.07710050650619224932e-11;

const S1 = -1.66666666666666324348e-1, S2 = 8.33333333332248946124e-3, S3 = -1.98412698298579493134e-4;
const S4 = 2.75573137070700676789e-6, S5 = -2.50507602534068634195e-8, S6 = 1.58969099521155010221e-10;
const C1 = 4.16666666666666019037e-2, C2 = -1.38888888888741095749e-3, C3 = 2.48015872894767294178e-5;
const C4 = -2.75573143513906633035e-7, C5 = 2.08757232129817482790e-9, C6 = -1.13596475577881948265e-11;

/** sin and cos on [-pi/4, pi/4]. */
function ksin(x: number): number {
  const z = x * x;
  return x + z * x * (S1 + z * (S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)))));
}
function kcos(x: number): number {
  const z = x * x;
  return 1 - (0.5 * z - z * z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6))))));
}

/** Sine of an angle in radians. Meant for the angles a game has (a few turns at most), where it is within 1e-15 of the true value. */
export function sin(x: number): number {
  const k = Math.round(x * INV_PIO2);
  const r = x - k * PIO2_HI - k * PIO2_LO;
  switch (k & 3) {
    case 0: return ksin(r);
    case 1: return kcos(r);
    case 2: return -ksin(r);
    default: return -kcos(r);
  }
}

/** Cosine of an angle in radians (see `sin`). */
export function cos(x: number): number {
  const k = Math.round(x * INV_PIO2);
  const r = x - k * PIO2_HI - k * PIO2_LO;
  switch (k & 3) {
    case 0: return kcos(r);
    case 1: return -ksin(r);
    case 2: return -kcos(r);
    default: return ksin(r);
  }
}

const ATAN_HI = [4.63647609000806093515e-1, 7.85398163397448278999e-1, 9.82793723247329054082e-1, 1.57079632679489655800];
const ATAN_LO = [2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17, 6.12323399573676603587e-17];
const A0 = 3.33333333333329318027e-1, A1 = -1.99999999998764832476e-1, A2 = 1.42857142725034663711e-1, A3 = -1.11111104054623557880e-1;
const A4 = 9.09088713343650656196e-2, A5 = -7.69187620504482999495e-2, A6 = 6.66107313738753120669e-2, A7 = -5.83357013379057348645e-2;
const A8 = 4.97687799461593236017e-2, A9 = -3.65315727442169155270e-2, A10 = 1.62858201153657823623e-2;
const PI_LO = 1.2246467991473531772e-16;

/** Arc tangent of a value >= 0. */
function atanPos(x: number): number {
  let id: number;
  if (x < 0.4375) {
    if (x < 1.8626451492309570e-9) return x;
    id = -1;
  } else if (x < 1.1875) {
    if (x < 0.6875) { id = 0; x = (2 * x - 1) / (2 + x); } else { id = 1; x = (x - 1) / (x + 1); }
  } else if (x < 2.4375) { id = 2; x = (x - 1.5) / (1 + 1.5 * x); } else if (x < 7.378697629483821e19) { id = 3; x = -1 / x; } else return ATAN_HI[3] + ATAN_LO[3];
  const z = x * x, w = z * z;
  const s1 = z * (A0 + w * (A2 + w * (A4 + w * (A6 + w * (A8 + w * A10)))));
  const s2 = w * (A1 + w * (A3 + w * (A5 + w * (A7 + w * A9))));
  if (id < 0) return x - x * (s1 + s2);
  return ATAN_HI[id] - (x * (s1 + s2) - ATAN_LO[id] - x);
}

/** The angle of the vector (x, y) in radians, in [-pi, pi], with the conventions of Math.atan2 (signed zeros included). */
export function atan2(y: number, x: number): number {
  if (x !== x || y !== y) return NaN;
  if (y === 0) {
    const neg = 1 / y < 0;
    if (x < 0 || (x === 0 && 1 / x < 0)) return neg ? -Math.PI : Math.PI;
    return y;
  }
  if (x === 0) return y < 0 ? -Math.PI / 2 : Math.PI / 2;
  const z = atanPos(Math.abs(y / x));
  if (x > 0) return y < 0 ? -z : z;
  return y < 0 ? z - PI_LO - Math.PI : Math.PI - (z - PI_LO);
}

/** Length of the vector (x, y). */
export function hypot(x: number, y: number): number {
  return Math.sqrt(x * x + y * y);
}

/** x squared (the `**` operator is not exact everywhere). */
export function sq(x: number): number {
  return x * x;
}
