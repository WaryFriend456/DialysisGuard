import './globals.css';
import { Inter, JetBrains_Mono } from 'next/font/google';
import { AuthProvider } from '@/contexts/AuthContext';
import { MonitoringProvider } from '@/contexts/MonitoringContext';
import { ThemeProvider } from '@/contexts/ThemeContext';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-ui',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-code',
});

export const metadata = {
  title: 'DialysisGuard - AI-Driven Hemodialysis Monitoring',
  description:
    'Real-time AI monitoring and adverse event prediction for hemodialysis with explainable AI',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" data-theme="dark">
      <body className={`${inter.variable} ${jetbrainsMono.variable}`}>
        <ThemeProvider>
          <AuthProvider>
            <MonitoringProvider>{children}</MonitoringProvider>
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
