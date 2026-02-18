import './globals.css';
import { AuthProvider } from '@/contexts/AuthContext';

export const metadata = {
  title: 'DialysisGuard — AI-Driven Hemodialysis Monitoring',
  description: 'Real-time AI monitoring and adverse event prediction for hemodialysis with Explainable AI',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
