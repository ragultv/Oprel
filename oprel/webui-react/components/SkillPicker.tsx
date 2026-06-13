import React, { useEffect, useRef, useState, useMemo } from "react";
import * as Icons from "lucide-react";
import { cn } from "@/services/utils";
import { useApp } from "@/services/context";
import { Skill, filterSkills, getMostUsedSkills } from "@/services/skills";

interface SkillPickerProps {
  searchQuery: string; // The search query typed after /
  onSelect: (skill: Skill) => void;
  onClose: () => void;
  className?: string;
}

export const SkillPicker: React.FC<SkillPickerProps> = ({
  searchQuery,
  onSelect,
  onClose,
  className
}) => {
  const { skills } = useApp();
  const [highlightedIndex, setHighlightedIndex] = useState<number>(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // If search query changes, reset selection index
  useEffect(() => {
    setHighlightedIndex(0);
  }, [searchQuery]);

  // Filter skills based on query and sort so most used appear first
  const visibleSkills = useMemo(() => {
    const filtered = filterSkills(searchQuery, skills);
    const mostUsedIds = getMostUsedSkills(skills).map(s => s.id);
    
    return [...filtered].sort((a, b) => {
      const aIdx = mostUsedIds.indexOf(a.id);
      const bIdx = mostUsedIds.indexOf(b.id);
      if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
      if (aIdx !== -1) return -1;
      if (bIdx !== -1) return 1;
      return 0;
    });
  }, [searchQuery, skills]);

  // Adjust highlighted index if it goes out of bounds
  useEffect(() => {
    if (highlightedIndex >= visibleSkills.length) {
      setHighlightedIndex(Math.max(0, visibleSkills.length - 1));
    }
  }, [visibleSkills.length, highlightedIndex]);

  // Dynamic Icon mapping for skills
  const renderSkillIcon = (iconName: string, category: string) => {
    const IconComponent = (Icons as any)[iconName] || Icons.HelpCircle;
    
    // Choose colors to match categories
    const colors: Record<string, string> = {
      Writing: "text-cyan-500 bg-cyan-500/10",
      Development: "text-indigo-500 bg-indigo-500/10",
      Research: "text-emerald-500 bg-emerald-500/10",
      Documents: "text-amber-500 bg-amber-500/10",
      Media: "text-rose-500 bg-rose-500/10",
    };
    const colorClass = colors[category] || "text-primary bg-primary/10";

    return (
      <div className={cn("w-7 h-7 rounded-lg flex items-center justify-center shrink-0 border border-white/[0.03]", colorClass)}>
        <IconComponent size={15} />
      </div>
    );
  };

  // Keyboard navigation logic
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (visibleSkills.length === 0) return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        setHighlightedIndex(prev => (prev + 1) % visibleSkills.length);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setHighlightedIndex(prev => (prev - 1 + visibleSkills.length) % visibleSkills.length);
      } else if (e.key === "Enter") {
        e.preventDefault();
        onSelect(visibleSkills[highlightedIndex]);
      } else if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };

    window.addEventListener("keydown", handleKeyDown, true);
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [visibleSkills, highlightedIndex, onSelect, onClose]);

  // Scroll active item into view
  useEffect(() => {
    if (!listRef.current) return;
    const activeItem = listRef.current.querySelector("[data-active='true']");
    if (activeItem) {
      activeItem.scrollIntoView({
        block: "nearest"
      });
    }
  }, [highlightedIndex]);

  // Close when clicking outside the picker container
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [onClose]);

  return (
    <div
      ref={containerRef}
      className={cn(
        "absolute bottom-full left-0 w-[480px] max-w-full mb-3 h-[360px] min-h-[360px] bg-[#1a1a1a]/95 backdrop-blur-md border border-border rounded-xl shadow-2xl flex flex-col overflow-hidden z-50 animate-in fade-in slide-in-from-bottom-2 duration-200",
        className
      )}
    >
      {/* Search header status indicator */}
      <div className="px-4 py-2 border-b border-border/40 flex items-center justify-between bg-[#151515]/20 select-none">
        <div className="text-[10px] text-muted-foreground font-semibold uppercase tracking-wider">
          {searchQuery ? `Searching for "${searchQuery}"` : "Select a skill"}
        </div>
        <div className="text-[9px] text-muted-foreground/60 flex items-center gap-1.5 font-medium">
          <kbd className="px-1 py-0.5 bg-neutral-800 border border-neutral-700 rounded text-[8px] font-mono leading-none">↑↓</kbd> Navigate
          <kbd className="px-1 py-0.5 bg-neutral-800 border border-neutral-700 rounded text-[8px] font-mono leading-none">Enter</kbd> Select
        </div>
      </div>

      {/* List items */}
      <div ref={listRef} className="flex-1 overflow-y-auto p-1.5 space-y-0.5 scroll-smooth bg-[#1a1a1a]/40">
        {visibleSkills.length > 0 ? (
          visibleSkills.map((skill, index) => {
            const isActive = index === highlightedIndex;
            return (
              <button
                key={skill.id}
                data-active={isActive}
                onClick={() => onSelect(skill)}
                onMouseEnter={() => setHighlightedIndex(index)}
                className={cn(
                  "w-full text-left p-2 rounded-lg transition-all duration-150 flex items-center gap-3 border cursor-pointer",
                  isActive
                    ? "bg-primary/10 text-foreground border-primary/20"
                    : "border-transparent text-muted-foreground hover:text-foreground bg-transparent"
                )}
              >
                {renderSkillIcon(skill.icon, skill.category)}
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className={cn("text-xs font-semibold truncate", isActive ? "text-foreground" : "text-foreground/80")}>
                      {skill.name}
                    </span>
                    <span className="text-[10px] opacity-40 font-mono">
                      /{skill.command}
                    </span>
                    {skill.isPremium && (
                      <span className="text-[8px] tracking-wide uppercase px-1 py-0.2 bg-primary/10 text-primary border border-primary/20 rounded font-bold scale-90">
                        PRO
                      </span>
                    )}
                  </div>
                  <p className="text-[10px] text-muted-foreground/80 truncate mt-0.5">
                    {skill.description}
                  </p>
                </div>
                
                {isActive && (
                  <Icons.CornerDownLeft size={11} className="text-primary opacity-80 shrink-0" />
                )}
              </button>
            );
          })
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground gap-2 select-none">
            <Icons.SearchCode size={24} className="opacity-30" />
            <div>
              <p className="text-xs font-semibold">No skills found</p>
              <p className="text-[10px] opacity-75 mt-0.5">Try searching for a different keyword</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
