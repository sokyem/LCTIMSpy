import { useEffect, useState } from "react";

import { useStore } from "../store";

/** Apply the chosen theme to <html> and report whether dark is in effect. */
export function useDarkMode(): boolean {
  const theme = useStore((s) => s.theme);
  const [systemDark, setSystemDark] = useState(() => matchMedia("(prefers-color-scheme: dark)").matches);

  useEffect(() => {
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setSystemDark(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
  }, [theme]);

  return theme === "dark" || (theme === "system" && systemDark);
}

/** Current value of a CSS custom property on <html>. */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
