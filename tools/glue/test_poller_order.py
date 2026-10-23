"""Compile the actual patched scan body against deterministic host request queues."""
import argparse
import pathlib
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('cpuinfer', type=pathlib.Path)
args = parser.parse_args()
source = args.cpuinfer.read_text(encoding='utf-8')
start = '        const size_t n = snap.size();\n'
end = '      }\n      __builtin_ia32_pause();'
assert source.count(start) == 1
body = start + source.split(start, 1)[1].split(end, 1)[0]
prefix = r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <functional>
#include <iostream>
#include <vector>
struct Entry { int idx; bool sync; int allow; bool permanent; std::function<void(int)> func; int args; };
using MemopEntry = Entry;
struct Queue { std::vector<int>& log; void sync(int allow) { log.push_back(1000+allow); } };
std::vector<int> run(int groups, size_t hint, bool arrive_mid_scan) {
  std::vector<int> log, retire;
  std::vector<Entry> snap;
  std::vector<int> req(groups*3,1), done(groups*3,0), reads(groups*3,0);
  for (int g=0;g<groups;++g) for(int j=0;j<3;++j) {
    int id=g*3+j;
    snap.push_back({id,j==2,g,j==1,[&](int arg){ log.push_back(arg); },id});
  }
  if(arrive_mid_scan) req[0]=0;
  auto req_host = [&](int i) { if(arrive_mid_scan && i==0 && ++reads[i]==2) req[i]=1; return &req[i]; };
  auto done_host = [&](int i) { return &done[i]; };
  Queue queue{log}; auto task_queue_=&queue;
'''
suffix = r'''
  assert(std::all_of(req.begin(),req.end(),[](int x){return x==0;}));
  assert(retire.size()==size_t(groups*2));
  for(int g=0;g<groups;++g) assert(done[g*3+2]==1);
  return log;
}
int main() {
  int checks=0;
  for(size_t hint=0;hint<3;++hint) { assert((run(1,hint,false)==std::vector<int>{0,1,1000})); ++checks; }
  assert((run(1,0,true)==std::vector<int>{0,1,1000})); ++checks;
  assert(run(0,7,false).empty()); ++checks;
  for(size_t hint=0;hint<6;++hint) {
    auto got=run(2,hint,false);
    assert(got.size()==6);
    auto pos=[&](int v){return std::find(got.begin(),got.end(),v)-got.begin();};
    assert(pos(0)<pos(1) && pos(1)<pos(1000));
    assert(pos(3)<pos(4) && pos(4)<pos(1001)); ++checks;
  }
  std::cout << "poller ordering: " << checks << " checks PASS\n";
}
'''
with tempfile.TemporaryDirectory(prefix='glue-poller-check-') as temp:
    root = pathlib.Path(temp)
    cpp, executable = root/'check.cpp', root/'check'
    cpp.write_text(prefix + body + suffix, encoding='utf-8')
    subprocess.run(['c++', '-std=c++17', '-O2', str(cpp), '-o', str(executable)], check=True, timeout=60)
    subprocess.run([str(executable)], check=True, timeout=10)
