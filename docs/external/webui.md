# Oprel Studio (WebUI)

Oprel Studio is a modern, React-based frontend application bundled with the SDK. It provides a rich ChatGPT-like interface for interacting with local models.

## Launching the UI

To start the UI along with the backend daemon:
```bash
oprel start
```
This spins up the FastAPI backend on port 11435, and then launches the Vite React server, automatically opening your browser to `http://127.0.0.1:11435/gui/` (or the configured port).

## Components & Architecture

The WebUI is located in `oprel/webui-react/`. It is built using **React**, **Vite**, and **TailwindCSS**.

### ChatView
The main interface (`components/ChatView.tsx`) manages the conversation history.
- It communicates directly with the Oprel daemon's `/api/chat` endpoint.
- It fully supports **streaming responses**, rendering Markdown and syntax-highlighted code blocks in real-time.
- It parses reasoning tokens `<think>...</think>` into collapsible accordion elements for models like DeepSeek-R1.

### OcrView
A dedicated tab (`components/OcrView.tsx`) for the OCR service.
- Allows users to drag-and-drop PDFs or images.
- Sends the file to the daemon's `/v1/ocr` endpoint.
- Renders bounding boxes dynamically over the original image using the normalized coordinates (`bbox_norm`) returned by the API.

### State Management
The UI uses lightweight React hooks and local storage to persist conversation histories on the client-side, syncing with the SQLite backend for persistent chat logs.
