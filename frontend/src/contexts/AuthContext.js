'use client';

import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { auth } from '@/lib/api';

const AuthContext = createContext(null);

export function dashboardPathForRole(role) {
    const paths = {
        super_admin: '/admin/organizations',
        org_admin: '/admin/staff',
        doctor: '/dashboard/doctor',
        nurse: '/dashboard/nurse',
    };
    return paths[role] || '/login';
}

function getStoredToken() {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem('token');
}

export function AuthProvider({ children }) {
    const initialToken = getStoredToken();
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(Boolean(initialToken));

    useEffect(() => {
        if (!initialToken) return;

        let cancelled = false;
        auth.me()
            .then((currentUser) => {
                if (!cancelled) {
                    setUser(currentUser);
                    if (currentUser.must_change_password && window.location.pathname !== '/change-password') {
                        window.location.href = '/change-password';
                    }
                }
            })
            .catch(() => {
                if (!cancelled) {
                    localStorage.removeItem('token');
                    setUser(null);
                }
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });

        return () => {
            cancelled = true;
        };
    }, [initialToken]);

    const login = async (email, password) => {
        const res = await auth.login({ email, password });
        localStorage.setItem('token', res.access_token);
        setUser(res.user);
        setLoading(false);
        return res.user;
    };

    const changePassword = async (currentPassword, newPassword) => {
        await auth.changePassword({ current_password: currentPassword, new_password: newPassword });
        const currentUser = await auth.me();
        setUser(currentUser);
        return currentUser;
    };

    const logout = () => {
        localStorage.removeItem('token');
        setUser(null);
        window.location.href = '/login';
    };

    const value = useMemo(() => ({
        user,
        loading,
        login,
        changePassword,
        logout,
    }), [loading, user]);

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within AuthProvider');
    }
    return context;
}
