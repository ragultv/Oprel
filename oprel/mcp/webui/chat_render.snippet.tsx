// Replaces the markdown rendering of a reply so tool calls show as collapsed groups.
// Needs: import { ToolCallGroup, groupToolCalls, splitToolCalls } from '@/components/ToolCallCard'

        {groupToolCalls(splitToolCalls(cleaned)).map((seg, i) =>
          seg.type === "tools" ? (
            <ToolCallGroup key={i} calls={seg.calls} />
          ) : seg.type === "text" && seg.text.trim() ? (
            <div key={i} className="oprel-markdown">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkBreaks]}
                rehypePlugins={[rehypeRaw]}
                components={renderers}
              >
                {seg.text}
              </ReactMarkdown>
            </div>
          ) : null
        )}
