#include <iostream>
using namespace std;
#define MAX 50

int dp[MAX][MAX];
int W[MAX] = { 0 };
int V[MAX] = { 0 };
int maxv = 0;
int x[MAX] = { 0 };
int n, w;

void dfs(int i, int tw, int tv, int op[])
{
    if (i > n)				//找到一个叶子结点
    {
        if (tw == w && tv > maxv)		//找到一个满足条件的更优解,保存 
        {
            maxv = tv;
            for (int j = 1; j <= n; j++)
                x[j] = op[j];
        }
    }
    else					//尚未找完所有物品
    {
        op[i] = 1;				//选取第i个物品
        dfs(i + 1, tw + W[i], tv + V[i], op);
        op[i] = 0;				//不选取第i个物品,回溯
        dfs(i + 1, tw, tv, op);
    }
}

int main()
{
    cout << "物品的个数n为：" << endl;
    cin >> n;
    cout << "背包的容量w为：" << endl;
    cin >> w;
    cout << "依次输入物品的重量为：" << endl;
    for (int i = 1; i < n + 1; i++)
    {
        cin >> W[i];
    }
    cout << "依次输入物品的价值为：" << endl;
    for (int i = 1; i < n + 1; i++)
    {
        cin >> V[i];
    }


    cout << "最大的总价值为：" << maxv << endl;
    cout << "背包中的物品为：";
    for (int i = 1; i <= n; i++)
    {
        if (x[i] != 0)
            cout << i << "  ";
    }
}
