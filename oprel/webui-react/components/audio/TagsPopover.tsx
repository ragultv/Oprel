"use client"

import React, { useState, useMemo } from "react"
import { Search, Sparkles, X } from "lucide-react"

export interface TagItem {
  tag: string
  label: string
  category: string
}

interface TagsPopoverProps {
  tags: TagItem[]
  onSelectTag: (tag: string) => void
  onClose: () => void
}

export function TagsPopover({ tags, onSelectTag, onClose }: TagsPopoverProps) {
  const [search, setSearch] = useState("")

  const filtered = useMemo(() => {
    if (!search.trim()) return tags
    const q = search.toLowerCase()
    return tags.filter(
      (t) => t.tag.toLowerCase().includes(q) || t.label.toLowerCase().includes(q) || t.category.toLowerCase().includes(q)
    )
  }, [tags, search])

  const categories = useMemo(() => {
    const map = new Map<string, TagItem[]>()
    for (const t of filtered) {
      if (!map.has(t.category)) map.set(t.category, [])
      map.get(t.category)!.push(t)
    }
    return Array.from(map.entries())
  }, [filtered])

  return (
    <>
      {/* Click outside backdrop to dismiss */}
      <div className="fixed inset-0 z-40 bg-transparent" onClick={onClose} />

      {/* Tooltip Card Anchored Above Trigger Button */}
      <div className="absolute bottom-full left-0 mb-3 w-[460px] sm:w-[520px] max-w-[calc(100vw-3rem)] bg-[#1a1a1a] border border-white/10 rounded-2xl shadow-2xl p-3.5 text-foreground z-50 animate-in fade-in zoom-in-95 duration-150 space-y-2.5 text-left">
        {/* Header */}
        <div className="flex items-center justify-between pb-2 border-b border-white/10">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-lg bg-primary/10 text-primary flex items-center justify-center border border-primary/20">
              <Sparkles size={12} />
            </div>
            <div>
              <h4 className="text-xs font-bold leading-tight text-foreground">Expressive Action Tags</h4>
              <p className="text-[10px] text-muted-foreground leading-tight">Click tag to insert at cursor</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/10 transition-all"
            title="Close"
          >
            <X size={14} />
          </button>
        </div>

        {/* Search Bar */}
        <div className="relative">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search tags (laughing, whisper, pause)..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-black/40 border border-white/10 rounded-xl py-1.5 pl-8 pr-3 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary/50 focus:outline-none transition-all"
            autoFocus
          />
        </div>

        {/* Tags by Category (Wider multi-column display) */}
        <div className="max-h-36 overflow-y-auto pr-1 space-y-2">
          {categories.length === 0 ? (
            <div className="text-center py-6 text-xs text-muted-foreground">No matching tags found</div>
          ) : (
            categories.map(([category, items]) => (
              <div key={category} className="space-y-1">
                <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider block px-0.5">
                  {category}
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {items.map((item) => (
                    <button
                      key={item.tag}
                      onClick={() => {
                        onSelectTag(item.tag)
                        onClose()
                      }}
                      className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-mono font-medium bg-white/5 border border-white/10 text-neutral-200 hover:bg-primary/20 hover:border-primary/40 hover:text-primary transition-all active:scale-95 shadow-xs"
                    >
                      <span>{item.tag}</span>
                    </button>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="pt-2 border-t border-white/10 flex items-center justify-between text-[10px] text-muted-foreground">
          <span>Fish Audio S1-mini</span>
          <span className="text-primary font-medium font-mono">{tags.length} Expressive Tags</span>
        </div>

        {/* Tooltip Downward Triangle Pointer */}
        <div className="absolute top-full left-8 -translate-x-1/2 border-[7px] border-transparent border-t-[#1a1a1a]" />
        <div className="absolute top-full left-8 -translate-x-1/2 border-[7px] border-transparent border-t-white/10 -z-10 translate-y-[1px]" />
      </div>
    </>
  )
}
