"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Cpu, Database, Sliders, PlayCircle, Activity } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();
  const [isBackendOnline, setIsBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch("/api/health");
        setIsBackendOnline(res.ok);
      } catch {
        setIsBackendOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { label: "Dashboard", href: "/", icon: Activity },
    { label: "1. Data Prep", href: "/data", icon: Database },
    { label: "2. Model Config", href: "/config", icon: Sliders },
    { label: "3. Train & Run", href: "/run", icon: PlayCircle },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-zinc-200 bg-white/80 backdrop-blur-md dark:border-zinc-800 dark:bg-zinc-950/80">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-2.5 font-bold text-xl tracking-tight text-zinc-900 dark:text-zinc-50">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600 text-white shadow-sm">
            <Cpu className="h-5 w-5" />
          </div>
          <span>Quick<span className="text-blue-600">Machine</span></span>
        </Link>

        <nav className="flex items-center gap-1 sm:gap-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all ${
                  isActive
                    ? "bg-blue-50 text-blue-700 dark:bg-blue-950/50 dark:text-blue-400 font-semibold"
                    : "text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs dark:border-zinc-800 dark:bg-zinc-900">
            <span
              className={`h-2 w-2 rounded-full ${
                isBackendOnline === true
                  ? "bg-emerald-500 animate-pulse"
                  : isBackendOnline === false
                  ? "bg-rose-500"
                  : "bg-amber-400"
              }`}
            />
            <span className="text-zinc-600 dark:text-zinc-400 font-medium">
              {isBackendOnline === true ? "API Online" : isBackendOnline === false ? "API Disconnected" : "Checking API..."}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
