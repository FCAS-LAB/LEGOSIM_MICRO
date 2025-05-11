#include"router.h"

using namespace std;
using namespace boost;

CRouter::CRouter(long port_cnt,long vc_cnt,long in_buf_size,long out_buf_size,const add_type&address,long ary_size,long flit_size):m_address(address)
{
	switch(configuration::ap().routing_alg()){
	case XY_:
		m_router=new CXYRouter(port_cnt,vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	case TXY_:
		m_router=new CTXYRouter(port_cnt,vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	case CHIPLET_ROUTING_MESH:
		m_router=new CChipletMeshRouter(port_cnt,vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	case CHIPLET_STAR_TOPO_ROUTING:
		m_router=new CChipletStarRouter(port_cnt,vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	case GRAPH_TOPO:
		m_router=new CGraphTopo(CGraphTopo::vPortCount[address.front()],vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	case RECONFIGURABLE_GRAPH_TOPO:
		m_router=new CReconfigTopoRouter(CGraphTopo::vPortCount[address.front()],vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	default:
		Sassert(false);
	}
}
CRouter::~CRouter()
{
	delete m_router;
}
CRouter&CRouter::operator=(const CRouter&r)
{
	copyFrom(r);
	return *this;
}
CRouter::CRouter(const CRouter&r)
{
	copyFrom(r);
}
void CRouter::copyFrom(const CRouter&r)
{
	delete m_router;
	long port_cnt,vc_cnt,in_buf_size,out_buf_size,ary_size,flit_size;
	const add_type&address=r.m_router->address();
	port_cnt=r.m_router->physic_ports();
	vc_cnt=r.m_router->vc_number();
	in_buf_size=r.m_router->buffer_size();
	out_buf_size=r.m_router->getOutBufferSize();
	ary_size=r.m_router->getArySize();
	flit_size=r.m_router->getFlitSize();
	switch(configuration::ap().routing_alg()){
	case XY_:
		m_router=new CXYRouter(port_cnt,vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	case TXY_:
		m_router=new CTXYRouter(port_cnt,vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	case CHIPLET_ROUTING_MESH:
		m_router=new CChipletMeshRouter(port_cnt,vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	case CHIPLET_STAR_TOPO_ROUTING:
		m_router=new CChipletStarRouter(port_cnt,vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	case GRAPH_TOPO:
		m_router=new CGraphTopo(port_cnt,vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	case RECONFIGURABLE_GRAPH_TOPO:
		m_router=new CReconfigTopoRouter(port_cnt,vc_cnt,in_buf_size,out_buf_size,address,ary_size,flit_size);
		break;
	default:
		Sassert(false);
	}
}
//https://zhuanlan.zhihu.com/p/347977300
CRouter::CRouter(CRouter&&r0)
{
	m_router=r0.m_router;
	r0.m_router=nullptr;
}
double CRouter::power_buffer_report()
{
	return m_router->power_buffer_report();
}
double CRouter::power_crossbar_report()
{
	return m_router->power_crossbar_report();
}
double CRouter::power_arbiter_report()
{
	return m_router->power_arbiter_report();
}
double CRouter::power_link_report()
{
	return m_router->power_link_report();
}
void CRouter::empty_check()const
{
	m_router->empty_check();
}
void CRouter::router_sim_pwr(time_type routingPeriod)
{
	m_router->router_sim_pwr(routingPeriod);
}
time_type CRouter::total_delay()const
{
	return m_router->total_delay();
}
void CRouter::receive_credit(long pc,long vc)
{
	m_router->receive_credit(pc,vc);
}
void CRouter::receive_packet()
{
	m_router->receive_packet();
}
void CRouter::receive_flit(long pc, long vc, flit_template&flit)
{
	m_router->receive_flit(pc,vc,flit);
}
void CRouter::inputTrace(const SPacket&packet)
{
	m_router->inputTrace(packet);
}
ostream&operator<<(ostream&os,const CRouter&router)
{
	return os<<(*router.m_router);
}

add_type CRouter::returnAddress(){
	return m_router->address();
}