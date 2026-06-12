import React from "react";
import * as Icons from "lucide-react";
import { cn } from "@/services/utils";
import { Skill } from "@/services/skills";

interface SkillChipProps {
  skill: Skill;
  onRemove?: () => void;
  className?: string;
}

// Category theme mapping for distinct styling
const CATEGORY_THEMES = {
  Writing: {
    bg: "bg-cyan-500/10 dark:bg-cyan-400/5",
    border: "border-cyan-500/30 dark:border-cyan-400/20",
    text: "text-cyan-600 dark:text-cyan-400",
    hoverBg: "hover:bg-cyan-500/20 dark:hover:bg-cyan-400/15"
  },
  Development: {
    bg: "bg-indigo-500/10 dark:bg-indigo-400/5",
    border: "border-indigo-500/30 dark:border-indigo-400/20",
    text: "text-indigo-600 dark:text-indigo-400",
    hoverBg: "hover:bg-indigo-500/20 dark:hover:bg-indigo-400/15"
  },
  Research: {
    bg: "bg-emerald-500/10 dark:bg-emerald-400/5",
    border: "border-emerald-500/30 dark:border-emerald-400/20",
    text: "text-emerald-600 dark:text-emerald-400",
    hoverBg: "hover:bg-emerald-500/20 dark:hover:bg-emerald-400/15"
  },
  Documents: {
    bg: "bg-amber-500/10 dark:bg-amber-400/5",
    border: "border-amber-500/30 dark:border-amber-400/20",
    text: "text-amber-600 dark:text-amber-400",
    hoverBg: "hover:bg-amber-500/20 dark:hover:bg-amber-400/15"
  },
  Media: {
    bg: "bg-rose-500/10 dark:bg-rose-400/5",
    border: "border-rose-500/30 dark:border-rose-400/20",
    text: "text-rose-600 dark:text-rose-400",
    hoverBg: "hover:bg-rose-500/20 dark:hover:bg-rose-400/15"
  }
};

export const SkillChip: React.FC<SkillChipProps> = ({ skill, onRemove, className }) => {
  // Dynamically load the icon from lucide-react
  const IconComponent = (Icons as any)[skill.icon] || Icons.Sparkles;
  const theme = CATEGORY_THEMES[skill.category] || CATEGORY_THEMES.Writing;

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs font-semibold shadow-sm select-none transition-all duration-200 animate-in fade-in zoom-in-95 duration-150 shrink-0",
        theme.bg,
        theme.border,
        theme.text,
        className
      )}
    >
      <IconComponent size={13} className="shrink-0" />
      <span>{skill.name}</span>
      {skill.isPremium && (
        <span className="text-[8px] tracking-wide uppercase px-1 py-0.2 bg-primary/15 text-primary rounded-md font-extrabold ml-0.5 border border-primary/20 scale-90">
          PRO
        </span>
      )}
      {onRemove && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          className={cn(
            "w-4 h-4 rounded-full flex items-center justify-center transition-colors ml-1 -mr-1",
            theme.hoverBg
          )}
          title={`Remove ${skill.name} skill`}
        >
          <Icons.X size={10} className="stroke-[2.5]" />
        </button>
      )}
    </div>
  );
};
