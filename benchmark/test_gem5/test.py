from m5.objects import *

# 创建系统对象
system = System()

# 设置时钟频率
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '1GHz'
system.clk_domain.voltage_domain = VoltageDomain()

# 创建CPU对象
system.cpu = TimingSimpleCPU()

# 配置内存
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('512MB')]

# 设置内存总线
system.membus = SystemXBar()

# 配置 L1 缓存
system.cpu.icache = L1ICache(size='32kB')
system.cpu.dcache = L1DCache(size='32kB')

# 连接 CPU 缓存与总线
system.cpu.icache_port = system.membus.slave
system.cpu.dcache_port = system.membus.slave

# 创建内存控制器
system.mem_ctrl = DDR3_1600_8x8()
system.mem_ctrl.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.master

# 系统端口
system.system_port = system.membus.slave

# 定义要模拟的程序 (process)
process = Process()

# 假设 Python 解释器位于系统中的 /usr/bin/python3
process.cmd = ['/home/qc/anaconda3/envs/gem5/bin/python3', 'hello.py']  # Python 解释器和脚本

# 将程序关联到 CPU
system.cpu.workload = process
system.cpu.createThreads()

# 创建根对象并开始模拟
root = Root(full_system = False, system = system)
m5.instantiate()

print("开始模拟 Python 脚本")
exit_event = m5.simulate()
print(f"模拟结束，退出 @ tick {m5.curTick()} 因为 {exit_event.getCause()}")
