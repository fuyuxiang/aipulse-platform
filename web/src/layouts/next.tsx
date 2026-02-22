import { Outlet } from 'umi';
import { SiderNav } from './sider-nav';


export default function NextLayout() {
  return (
    <main className="flex h-screen overflow-hidden">
      <SiderNav />
      <section className="flex-1 flex flex-col overflow-hidden">
        <Outlet />
      </section>
    </main>
  );
}
