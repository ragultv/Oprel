# Canvas

The Canvas is an interactive artifact viewer built into Oprel Studio, heavily inspired by Anthropic's Artifacts feature.

## Purpose

When you ask an LLM to generate a substantial piece of content—such as a React component, a Python script, an HTML page, or a long-form essay—reading it in a narrow chat bubble is inefficient. The Canvas automatically extracts this content and renders it in a dedicated, resizable split-pane.

## How it Works

1. **Detection:** The `ChatView` component scans incoming streamed text for specific markdown code block fences (e.g., ` ```python ` or ` ```html `).
2. **Extraction:** If a code block exceeds a certain length, or if the model uses specific XML tags (if configured), the UI flags it as an "Artifact".
3. **Rendering (`components/Canvas.tsx`):** The Canvas component takes over.
   - For **Code**: It renders a full-fledged Monaco Editor (the same engine behind VS Code) with syntax highlighting, line numbers, and a "Copy to Clipboard" button.
   - For **Markdown/Essays**: It renders a rich-text document view.
   - For **HTML/React**: (In advanced modes) It can render the code inside a sandboxed iframe to provide a live preview of the generated UI.

## Interactive Editing
The Canvas is not read-only. Users can manually edit the code inside the Monaco editor. If they then ask a follow-up question in the chat (e.g., "Change the background to red"), the UI automatically appends the *current* state of the Canvas to the context, ensuring the model knows about any manual edits the user made.
