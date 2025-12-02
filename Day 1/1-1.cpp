#include <bits/stdc++.h>
using namespace std;

int main() {
    char c;
    int cur = 50, ans = 0;

    while(cin >> c) {
        int num;
        cin >> num;
        
        while(num--) {
            if(c == 'L') cur = (cur - 1 + 100) % 100;
            else cur = (cur + 1) % 100;
        }
        
        if(cur == 0) ++ans;
    }

    cout << ans;
}