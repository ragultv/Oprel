import ts from "typescript";

/**
 * Mechanical repairs for the API slips small models make over and over (and keep making after being told).
 *
 * Driven by the TypeScript compiler's own diagnostics: a rule only fires on the exact node the compiler flagged, rewrites it to the
 * documented form, and the result is type-checked again. Nothing here guesses at intent beyond the shape of the call.
 */

export type RawDiagnostic = { code: number; start: number; length: number; message: string };
export type Fix = { rule: string; line: number; detail: string };

type Edit = { from: number; to: number; text: string; fix: Fix };

const MOTION_HELPERS = new Set(["spring", "interpolate", "ease", "random"]);
const ANCHOR: Record<string, string> = { center: "middle", middle: "middle", left: "start", start: "start", right: "end", end: "end" };

function innermost<T extends ts.Node>(sf: ts.SourceFile, start: number, end: number, pred: (n: ts.Node) => n is T): T | undefined {
  let found: T | undefined;
  const visit = (n: ts.Node) => {
    if (n.getStart(sf) <= start && n.end >= end) {
      if (pred(n)) found = n;
      ts.forEachChild(n, visit);
    }
  };
  visit(sf);
  return found;
}

const calleeName = (c: ts.CallExpression): string | undefined => (ts.isIdentifier(c.expression) ? c.expression.text : undefined);

/** Compute non-overlapping edits for the diagnostics and return the repaired source plus what was changed. */
export function autofixScene(source: string, diagnostics: RawDiagnostic[]): { source: string; fixes: Fix[] } {
  const sf = ts.createSourceFile("scene.tsx", source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX);
  const lineOf = (pos: number) => sf.getLineAndCharacterOfPosition(pos).line + 1;
  const edits: Edit[] = [];

  for (const d of diagnostics) {
    const end = d.start + d.length;

    // interpolate(value, inputRange) -> interpolate(value, inputRange, [0, 1], { extrapolateRight: "clamp" })
    // spring(frame, fps) -> spring({ frame, fps })       ease(frame, start) -> ease(frame, start, start + 30)
    if (d.code === 2554 || d.code === 2555) {
      const call = innermost(sf, d.start, end, ts.isCallExpression);
      const name = call && calleeName(call);
      if (call && name === "interpolate" && call.arguments.length === 2) {
        const last = call.arguments[1];
        edits.push({ from: last.end, to: last.end, text: ', [0, 1], { extrapolateRight: "clamp" }',
          fix: { rule: "interpolate-arity", line: lineOf(call.getStart(sf)), detail: "interpolate() needs an output range: added [0, 1] and clamping" } });
      } else if (call && name === "spring" && call.arguments.length >= 2 && !ts.isObjectLiteralExpression(call.arguments[0])) {
        const [frame, fps, config] = call.arguments;
        const cfg = config && ts.isObjectLiteralExpression(config) ? `, config: ${config.getText(sf)}` : "";
        edits.push({ from: call.getStart(sf), to: call.end, text: `spring({ frame: ${frame.getText(sf)}, fps: ${fps.getText(sf)}${cfg} })`,
          fix: { rule: "spring-signature", line: lineOf(call.getStart(sf)), detail: "spring() takes one object: { frame, fps, config }" } });
      } else if (call && name === "random" && call.arguments.length === 3) {
        const [seed, lo, hi] = call.arguments;
        edits.push({ from: call.getStart(sf), to: call.end, text: `(${lo.getText(sf)} + random(${seed.getText(sf)}) * (${hi.getText(sf)} - ${lo.getText(sf)}))`,
          fix: { rule: "random-range", line: lineOf(call.getStart(sf)), detail: "random(seed) takes only a seed: scaled the result into the requested range" } });
      } else if (call && name === "ease" && call.arguments.length === 2) {
        const [f, s] = call.arguments;
        edits.push({ from: call.getStart(sf), to: call.end, text: `ease(${f.getText(sf)}, ${s.getText(sf)}, ${s.getText(sf)} + 30)`,
          fix: { rule: "ease-arity", line: lineOf(call.getStart(sf)), detail: "ease(frame, start, end) needs an end frame: used start + 30" } });
      }
    }

    // spring({...})(100)  ->  (spring({...}) * (100))   (these helpers return plain numbers, they are not functions)
    if (d.code === 2349) {
      const flagged = innermost(sf, d.start, end, ts.isCallExpression);
      // the error span is the callee (the inner spring(...) call); the node to rewrite is the call applied to it
      const call = flagged && ts.isCallExpression(flagged.parent) && flagged.parent.expression === flagged ? flagged.parent : flagged;
      if (call && ts.isCallExpression(call.expression)) {
        const inner = call.expression;
        const name = calleeName(inner);
        if (name && MOTION_HELPERS.has(name)) {
          const arg = call.arguments[0];
          edits.push({ from: call.getStart(sf), to: call.end, text: arg ? `(${inner.getText(sf)} * (${arg.getText(sf)}))` : inner.getText(sf),
            fix: { rule: "number-called-as-function", line: lineOf(call.getStart(sf)), detail: `${name}() returns a number: multiplied instead of calling it` } });
        }
      }
    }

    // spring(...).value / interpolate(...).value -> no .value
    if ((d.code === 2551 || d.code === 2339) && /'value'/.test(d.message)) {
      const access = innermost(sf, d.start, end, ts.isPropertyAccessExpression);
      if (access && access.name.text === "value") {
        edits.push({ from: access.getStart(sf), to: access.end, text: access.expression.getText(sf),
          fix: { rule: "no-dot-value", line: lineOf(access.getStart(sf)), detail: "animation helpers return plain numbers: removed .value" } });
      }
    }

    // <text textAlign="center"> -> textAnchor="middle";  <Backdrop bg=...> -> drop the prop that component does not have
    if (d.code === 2322) {
      const attr = innermost(sf, d.start, end, ts.isJsxAttribute);
      if (attr && ts.isIdentifier(attr.name)) {
        const name = attr.name.text;
        const tag = ts.isJsxOpeningElement(attr.parent.parent) || ts.isJsxSelfClosingElement(attr.parent.parent) ? attr.parent.parent.tagName.getText(sf) : "";
        if (name === "textAlign" && (tag === "text" || tag === "tspan")) {
          const raw = attr.initializer && ts.isStringLiteral(attr.initializer) ? attr.initializer.text : "middle";
          edits.push({ from: attr.getStart(sf), to: attr.end, text: `textAnchor="${ANCHOR[raw] ?? "middle"}"`,
            fix: { rule: "svg-text-anchor", line: lineOf(attr.getStart(sf)), detail: "SVG text uses textAnchor, not textAlign" } });
        } else if (/Property '\w+' does not exist on type 'IntrinsicAttributes/.test(d.message)) {
          edits.push({ from: attr.getFullStart(), to: attr.end, text: "",
            fix: { rule: "unknown-prop", line: lineOf(attr.getStart(sf)), detail: `<${tag}> has no "${name}" prop: removed it` } });
        }
      }
    }
  }

  // drop overlapping edits (keep the first), apply from the end so offsets stay valid
  edits.sort((a, b) => a.from - b.from);
  const kept: Edit[] = [];
  for (const e of edits) if (!kept.length || e.from >= kept[kept.length - 1].to) kept.push(e);
  let out = source;
  for (const e of [...kept].reverse()) out = out.slice(0, e.from) + e.text + out.slice(e.to);
  return { source: out, fixes: kept.map((e) => e.fix) };
}
