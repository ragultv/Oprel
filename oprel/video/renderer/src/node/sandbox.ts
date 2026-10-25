import ts from "typescript";
import type { Issue } from "../compiler";

/**
 * Static policy for scene components written by the AI agent (or a user).
 *
 * This is defence in depth, not the only barrier: the render browser also runs under a Content-Security-Policy that blocks
 * all network access (see render.ts). The checks here exist to (1) refuse obviously unsafe code before it is ever bundled and
 * (2) enforce the frame-determinism rules, with messages written so a model can fix its own code.
 */

export const ALLOWED_IMPORTS = new Set([
  "react",
  "react/jsx-runtime",
  "remotion",
  "@remotion/three",
  "@remotion/paths",
  "three",
  "@oprel/kit",
]);

const MAX_SOURCE_BYTES = 60_000;

/** Globals that are never allowed, whether or not the file also declares something with that name. */
const BANNED_IDENTIFIERS: Record<string, string> = {
  fetch: "network access is not allowed",
  XMLHttpRequest: "network access is not allowed",
  WebSocket: "network access is not allowed",
  EventSource: "network access is not allowed",
  Worker: "workers are not allowed",
  SharedWorker: "workers are not allowed",
  ServiceWorker: "workers are not allowed",
  importScripts: "loading scripts is not allowed",
  eval: "eval is not allowed",
  Function: "the Function constructor is not allowed",
  setTimeout: "timers are not frame-driven; derive everything from useCurrentFrame()",
  setInterval: "timers are not frame-driven; derive everything from useCurrentFrame()",
  requestAnimationFrame: "use useCurrentFrame() instead; wall-clock animation breaks rendering",
  requestIdleCallback: "use useCurrentFrame() instead",
  localStorage: "browser storage is not allowed",
  sessionStorage: "browser storage is not allowed",
  indexedDB: "browser storage is not allowed",
  process: "Node APIs are not available",
  require: "use ES imports from the allowed list",
  Date: "wall-clock time breaks deterministic rendering; derive time from useCurrentFrame() / fps",
  performance: "wall-clock time breaks deterministic rendering; use useCurrentFrame()",
  crypto: "use random(seed) from remotion for seeded randomness",
  navigator: "browser APIs are not allowed",
  document: "do not touch the DOM; return JSX",
  window: "use useVideoConfig() for width/height; do not use window",
  globalThis: "global object access is not allowed",
  self: "global object access is not allowed",
  useFrame: "useFrame() is wall-clock driven; compute 3D transforms from useCurrentFrame()",
};

/** Browser globals that are only a problem when they refer to the global (i.e. the file does not declare its own `top`, `open`...). */
const BANNED_GLOBALS_IF_UNDECLARED: Record<string, string> = {
  top: "global object access is not allowed",
  parent: "global object access is not allowed",
  frames: "global object access is not allowed",
  open: "opening windows is not allowed",
  alert: "dialogs are not allowed",
  confirm: "dialogs are not allowed",
  prompt: "dialogs are not allowed",
  module: "use ES exports",
  name: "global object access is not allowed",
  status: "global object access is not allowed",
  location: "navigation is not allowed",
  history: "navigation is not allowed",
};

const BANNED_PROPERTIES: Record<string, string> = {
  constructor: "prototype tricks are not allowed",
  __proto__: "prototype tricks are not allowed",
  dangerouslySetInnerHTML: "raw HTML injection is not allowed",
  random: "", // handled specially for Math.random
};

const BANNED_JSX = new Set(["script", "iframe", "object", "embed", "link", "meta", "base", "form", "input", "textarea", "audio", "video"]);
/** SVG SMIL animation runs on the browser clock, so it never advances frame by frame: scenes using it render as static images. */
const TIME_BASED_SVG = new Set(["animate", "animateTransform", "animateMotion", "set", "animateColor"]);
const CSS_MOTION_KEYS = new Set(["animation", "animationName", "animationDuration", "transition", "transitionDuration", "transitionProperty"]);
const URL_RE = /^\s*(?:[a-z][a-z0-9+.-]*:\/\/|\/\/|javascript:|data:text\/html)/i;
const SAFE_URL_PREFIXES = ["http://www.w3.org/", "https://www.w3.org/"];

export function checkSceneSource(source: string, fileName = "scene.tsx"): Issue[] {
  const issues: Issue[] = [];
  const err = (node: ts.Node | null, message: string, sf?: ts.SourceFile) => {
    let loc = fileName;
    if (node && sf) {
      const { line } = sf.getLineAndCharacterOfPosition(node.getStart(sf));
      loc = `${fileName}:${line + 1}`;
    }
    issues.push({ level: "error", path: loc, message });
  };

  if (Buffer.byteLength(source) > MAX_SOURCE_BYTES) {
    err(null, `file is larger than ${MAX_SOURCE_BYTES} bytes; split the idea into a smaller component`);
    return issues;
  }

  const sf = ts.createSourceFile(fileName, source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX);
  const syntax = (sf as unknown as { parseDiagnostics?: ts.Diagnostic[] }).parseDiagnostics ?? [];
  for (const d of syntax) {
    const { line } = sf.getLineAndCharacterOfPosition(d.start ?? 0);
    issues.push({ level: "error", path: `${fileName}:${line + 1}`, message: `syntax error: ${ts.flattenDiagnosticMessageText(d.messageText, " ")}` });
  }

  let hasDefaultExport = false;

  // names the file declares itself (variables, params, functions, imports, destructuring)
  const declared = new Set<string>();
  const collect = (node: ts.Node): void => {
    if ((ts.isVariableDeclaration(node) || ts.isParameter(node) || ts.isFunctionDeclaration(node) || ts.isClassDeclaration(node) || ts.isImportSpecifier(node) || ts.isBindingElement(node)) && node.name && ts.isIdentifier(node.name)) declared.add(node.name.text);
    if (ts.isImportClause(node) && node.name) declared.add(node.name.text);
    ts.forEachChild(node, collect);
  };
  collect(sf);

  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node) || (ts.isExportDeclaration(node) && node.moduleSpecifier)) {
      const spec = (node.moduleSpecifier as ts.StringLiteral).text;
      if (!ALLOWED_IMPORTS.has(spec)) err(node, `import "${spec}" is not allowed. Allowed modules: ${[...ALLOWED_IMPORTS].filter((m) => m !== "react/jsx-runtime").join(", ")}`, sf);
      if (spec === "@react-three/fiber") err(node, "do not import @react-three/fiber; use ThreeCanvas from @remotion/three", sf);
    }
    if (ts.isImportEqualsDeclaration(node)) err(node, "import = require() is not allowed", sf);
    if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) err(node, "dynamic import() is not allowed", sf);

    if (ts.isExportAssignment(node) && !node.isExportEquals) hasDefaultExport = true;
    if ((ts.isFunctionDeclaration(node) || ts.isClassDeclaration(node)) && node.modifiers?.some((m) => m.kind === ts.SyntaxKind.DefaultKeyword)) hasDefaultExport = true;
    if (ts.isExportDeclaration(node) && node.exportClause && ts.isNamedExports(node.exportClause)) {
      if (node.exportClause.elements.some((e) => e.name.text === "default")) hasDefaultExport = true;
    }

    if (ts.isIdentifier(node)) {
      const parent = node.parent;
      const isMemberName = ts.isPropertyAccessExpression(parent) && parent.name === node;
      const isPropKey =
        (ts.isPropertyAssignment(parent) || ts.isPropertySignature(parent) || ts.isMethodDeclaration(parent) || ts.isJsxAttribute(parent) || ts.isPropertyDeclaration(parent)) &&
        (parent as { name?: ts.Node }).name === node;
      const inType = ts.isTypeReferenceNode(parent) || ts.isQualifiedName(parent) || ts.isTypeQueryNode(parent) || ts.isTypeAliasDeclaration(parent) || ts.isInterfaceDeclaration(parent);
      const isShorthandKey = ts.isShorthandPropertyAssignment(parent);
      if (!isMemberName && !isPropKey && !inType) {
        if (BANNED_IDENTIFIERS[node.text] !== undefined) err(node, `"${node.text}" is not allowed: ${BANNED_IDENTIFIERS[node.text]}`, sf);
        else if (BANNED_GLOBALS_IF_UNDECLARED[node.text] !== undefined && !declared.has(node.text) && !isShorthandKey)
          err(node, `"${node.text}" is not allowed: ${BANNED_GLOBALS_IF_UNDECLARED[node.text]}`, sf);
      }
    }

    if (ts.isPropertyAccessExpression(node)) {
      const name = node.name.text;
      if (name === "constructor" || name === "__proto__") err(node, `".${name}" is not allowed: prototype tricks are not allowed`, sf);
      if (name === "random" && ts.isIdentifier(node.expression) && node.expression.text === "Math")
        err(node, `Math.random() is not deterministic; use random("seed") from remotion`, sf);
      if (name === "dangerouslySetInnerHTML") err(node, "raw HTML injection is not allowed", sf);
    }
    if (ts.isElementAccessExpression(node)) {
      const arg = node.argumentExpression;
      if (ts.isStringLiteralLike(arg) && (BANNED_PROPERTIES[arg.text] || BANNED_IDENTIFIERS[arg.text]))
        err(node, `["${arg.text}"] access is not allowed`, sf);
      else if (!ts.isStringLiteralLike(arg) && !ts.isNumericLiteral(arg) && !ts.isIdentifier(arg) && !ts.isPrefixUnaryExpression(arg) && !ts.isBinaryExpression(arg) && !ts.isTemplateExpression(arg) && !ts.isPropertyAccessExpression(arg) && !ts.isCallExpression(arg) && !ts.isConditionalExpression(arg) && !ts.isParenthesizedExpression(arg) && !ts.isElementAccessExpression(arg) && !ts.isNonNullExpression(arg) && !ts.isAsExpression(arg))
        err(node, "unsupported computed property access", sf);
    }

    if (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) {
      const tag = node.tagName.getText(sf);
      if (TIME_BASED_SVG.has(tag))
        err(node, `<${tag}> is time-based and does not render frame by frame (the scene would look static). Compute the attribute from the frame instead, e.g. cx={interpolate(frame, [0, 60], [100, 400], { extrapolateRight: "clamp" })}`, sf);
      else if (BANNED_JSX.has(tag)) err(node, `<${tag}> is not allowed in scenes (no scripts, frames, embeds, forms or media elements)`, sf);
    }
    if (ts.isJsxAttribute(node) && node.name.getText(sf) === "dangerouslySetInnerHTML") err(node, "raw HTML injection is not allowed", sf);

    if (ts.isJsxExpression(node) && node.expression && (ts.isJsxElement(node.parent) || ts.isJsxFragment(node.parent))) {
      const expText = node.expression.getText(sf).trim();
      if (expText === "narration" || expText === "props.narration" || expText.startsWith("narration.split") || expText.startsWith("narration.slice") || expText.startsWith("narration.substring")) {
        err(node, "do not render the narration prop as text/subtitles; synchronized captions are rendered globally by the video player. Remove {narration} from JSX.", sf);
      }
    }

    if ((ts.isPropertyAssignment(node) || ts.isShorthandPropertyAssignment(node)) && (ts.isIdentifier(node.name) || ts.isStringLiteralLike(node.name))) {
      if (CSS_MOTION_KEYS.has(node.name.text))
        err(node, `CSS "${node.name.text}" is time-based and renders inconsistently; animate with interpolate()/spring() on the current frame`, sf);
    }

    if (ts.isStringLiteralLike(node) && URL_RE.test(node.text) && !SAFE_URL_PREFIXES.some((p) => node.text.startsWith(p))) {
      if (!(ts.isImportDeclaration(node.parent) || ts.isExportDeclaration(node.parent)))
        err(node, `external URL "${node.text.slice(0, 60)}" is not allowed; scenes may not load anything from the network`, sf);
    }
    if (ts.isNoSubstitutionTemplateLiteral(node) || ts.isTemplateHead(node)) {
      if (URL_RE.test(node.text)) err(node, "external URLs are not allowed in templates", sf);
    }

    ts.forEachChild(node, visit);
  };
  visit(sf);

  if (!hasDefaultExport) err(null, "the file must `export default` the scene component (a function taking SceneProps)");
  return issues;
}
