import { redirect } from 'next/navigation';

export default function CaregiverRedirect() {
    redirect('/dashboard/nurse');
}
