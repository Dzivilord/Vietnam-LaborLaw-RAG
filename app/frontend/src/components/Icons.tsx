/** Small inline icons (stroke = currentColor) so no icon dependency is needed. */
type IconProps = { size?: number; className?: string };

function base(size: number) {
  return {
    width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
    strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true,
  };
}

export function SendIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M5 12h13" /><path d="m12 5 7 7-7 7" /></svg>;
}
export function PlusIcon({ size = 16, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M12 5v14M5 12h14" /></svg>;
}
export function BookIcon({ size = 16, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z" /><path d="M4 20.5A2.5 2.5 0 0 1 6.5 18H20v3H6.5" /></svg>;
}
export function CloseIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M6 6l12 12M18 6 6 18" /></svg>;
}
export function CopyIcon({ size = 15, className }: IconProps) {
  return <svg {...base(size)} className={className}><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V6a2 2 0 0 1 2-2h9" /></svg>;
}
export function CheckIcon({ size = 15, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="m5 12 5 5 9-10" /></svg>;
}
export function RetryIcon({ size = 15, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M4 12a8 8 0 1 0 2.4-5.7" /><path d="M4 4v4h4" /></svg>;
}
export function AlertIcon({ size = 16, className }: IconProps) {
  return <svg {...base(size)} className={className}><circle cx="12" cy="12" r="9" /><path d="M12 7.5v5.5M12 16.5h.01" /></svg>;
}
export function ScaleIcon({ size = 20, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M12 3v18M7 21h10M5 7h14" /><path d="M5 7l-3 6a3 3 0 0 0 6 0z" /><path d="M19 7l-3 6a3 3 0 0 0 6 0z" />
    </svg>
  );
}
