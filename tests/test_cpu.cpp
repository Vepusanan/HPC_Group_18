#define LIFE_NO_MAIN
#include "../cpu/game_of_life_cpu.cpp"
#include <cassert>

// Independent reference: scatter each live cell's contribution to neighbours.
std::vector<int> reference(const std::vector<int>& a, int n) {
    std::vector<int> counts(n*n), out(n*n);
    for (int r=0;r<n;++r) for (int c=0;c<n;++c) if(a[r*n+c])
        for(int y=r-1;y<=r+1;++y) for(int x=c-1;x<=c+1;++x)
            if(y>=0 && y<n && x>=0 && x<n && (y!=r || x!=c)) ++counts[y*n+x];
    for(int i=0;i<n*n;++i) out[i]=counts[i]==3 || (a[i] && counts[i]==2);
    return out;
}
int main() {
    for(int alive=0;alive<=1;++alive) for(int neighbours=0;neighbours<=8;++neighbours)
        assert(nextCellState(alive,neighbours)==(neighbours==3 || (alive && neighbours==2)));
    // Stable block on the corner verifies edges remain active (only exterior is dead).
    std::vector<int> block={1,1,0,1,1,0,0,0,0}, next(9);
    updateGrid(block.data(),next.data(),3); assert(next==block);
    std::vector<int> blink(25), expected(25), output(25);
    blink[11]=blink[12]=blink[13]=1;
    expected[7]=expected[12]=expected[17]=1;
    updateGrid(blink.data(),output.data(),5); assert(output==expected);
    updateGrid(output.data(),expected.data(),5); assert(expected==blink);
    for(int n: {1,2,3,15,16,17,31,32,33,65}) {
        std::vector<int> a(n*n), b(n*n);
        initializeGrid(a.data(),n);
        auto ref=a;
        int* current=a.data(); int* scratch=b.data();
        for(int generation=0;generation<100;++generation) {
            ref=reference(ref,n);
            runSimulation(current,scratch,n,1);
            assert(std::equal(ref.begin(),ref.end(),current));
        }
    }
    std::cout << "CPU tests: PASSED (rules, patterns, boundaries, 100-generation reference)\n";
}
