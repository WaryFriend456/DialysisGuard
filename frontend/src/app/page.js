'use client';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { dashboardPathForRole, useAuth } from '@/contexts/AuthContext';
import { Loader2 } from 'lucide-react';

export default function Home() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading) {
      if (user) {
        router.push(user.must_change_password ? '/change-password' : dashboardPathForRole(user.role));
      } else {
        router.push('/login');
      }
    }
  }, [user, loading, router]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-bg-primary">
      <Loader2 className="h-8 w-8 animate-spin text-accent" />
    </div>
  );
}
