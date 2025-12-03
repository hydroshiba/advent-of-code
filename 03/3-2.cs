using System;

class AdventOfCode {
	public static void Main(string[] args) {
		string line;
		
		long ans = 0;

		while((line = Console.ReadLine()) != null) {
			int[] arr = line.Select(c => (int)c - 48).ToArray();
			int n = arr.Length;
			int k = 12;

			// DP[j, i] holds the largest value constructible from j digits
			// that ends exactly at index i

			long[][] dp = new long[k + 1][];
			dp[0] = new long[n];
			
			for(int j = 1; j <= k; ++j) {
				dp[j] = new long[n];
				long prev_max = 0;

				for(int i = 0; i < n; ++i) {
					dp[j][i] = Math.Max(dp[j][i], prev_max * 10 + arr[i]);
					prev_max = Math.Max(prev_max, dp[j - 1][i]);
				}
			}

			// Iterate all indices at the kth row and take the max constructible
			// value from k digits
			long val = dp[k].Max();
			ans += val;
		}

		Console.WriteLine(ans);
	}
}