#ifndef ROUTER_H
#define ROUTER_H

//#include"../sim_router.h"
#include"reconfigurable_topo.h"
//#include"global.h"

class CRouter
{
	sim_router_template*m_router=nullptr;
public:
	CRouter(long port_cnt,long vc_cnt,long in_buf_size,long out_buf_size,const add_type&address,long ary_size,long flit_size);
	~CRouter();
	CRouter&operator=(const CRouter&r);
	CRouter(const CRouter&r);
	void copyFrom(const CRouter&r);
	//移动构造函数
	CRouter(CRouter&&r0);
	add_type m_address;
	double power_buffer_report();
	double power_crossbar_report();
	double power_arbiter_report();
	double power_link_report();
	void empty_check()const;
	void router_sim_pwr(time_type routingPeriod);
	time_type total_delay()const;
	void receive_credit(long pc, long vc);
	void receive_packet();
	void receive_flit(long pc, long vc, flit_template&flit);
	void inputTrace(const SPacket&packet);
	add_type returnAddress();
	friend ostream&operator<<(ostream&os,const CRouter&router);
};
ostream&operator<<(ostream&os,const CRouter&router);

#endif
