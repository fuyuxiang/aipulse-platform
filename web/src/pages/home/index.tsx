import { Applications } from './applications';
import { NextBanner } from './banner';
import { Datasets } from './datasets';

const Home = () => {
  return (
    <section style={{ background: 'rgba(248, 248, 248, 1)', height: '100%' }}>
      {/* <NextBanner></NextBanner> */}
      <section className="px-10 py-10 overflow-auto " style={{background:'rgba(248, 248, 248, 1)',height:'100%'}}>
        <Datasets></Datasets>
        <Applications></Applications>
      </section>
    </section>
  );
};

export default Home;
