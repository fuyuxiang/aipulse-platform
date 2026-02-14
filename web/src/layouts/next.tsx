import { Outlet } from 'umi';
import { Header } from './next-header';


export default function NextLayout() {
  return (
    <main className="flex flex-col h-full">
      <Header />
      <Outlet />
    </main>
  );
}
