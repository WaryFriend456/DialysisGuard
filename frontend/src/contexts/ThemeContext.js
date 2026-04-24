'use client';

import { createContext, useContext, useEffect, useMemo, useState } from 'react';

const ThemeContext = createContext(null);
const STORAGE_KEY = 'dialysisguard_theme_preference';

function getStoredPreference() {
    if (typeof window === 'undefined') return 'system';
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === 'light' || raw === 'dark' || raw === 'system') {
        return raw;
    }
    return 'system';
}

export function ThemeProvider({ children }) {
    const [preference, setPreference] = useState(() => getStoredPreference());
    const [systemPrefersDark, setSystemPrefersDark] = useState(() => {
        if (typeof window === 'undefined') return true;
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    });

    const resolvedTheme = useMemo(() => {
        if (preference === 'light' || preference === 'dark') return preference;
        return systemPrefersDark ? 'dark' : 'light';
    }, [preference, systemPrefersDark]);

    useEffect(() => {
        if (typeof window === 'undefined') return;
        localStorage.setItem(STORAGE_KEY, preference);
    }, [preference]);

    useEffect(() => {
        if (typeof window === 'undefined') return;

        const media = window.matchMedia('(prefers-color-scheme: dark)');
        const handleSystemThemeChange = () => {
            setSystemPrefersDark(media.matches);
        };
        media.addEventListener('change', handleSystemThemeChange);
        return () => media.removeEventListener('change', handleSystemThemeChange);
    }, []);

    useEffect(() => {
        if (typeof document === 'undefined') return;
        document.documentElement.setAttribute('data-theme', resolvedTheme);
        document.documentElement.style.colorScheme = resolvedTheme;
    }, [resolvedTheme]);

    const value = useMemo(
        () => ({
            preference,
            resolvedTheme,
            setPreference,
        }),
        [preference, resolvedTheme],
    );

    return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within ThemeProvider');
    }
    return context;
}
