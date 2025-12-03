using System;

class AdventOfCode {
	public static void Main(string[] args) {
		string line;
		
		int ans = 0;

		while((line = Console.ReadLine()) != null) {
			int[] arr = line.Select(c => (int)c - 48).ToArray();
			int max = 0, val = 0;

			for(int i = 0; i < arr.Length; ++i) {
				val = Math.Max(val, max * 10 + arr[i]);
				max = Math.Max(max, arr[i]);
			}

			ans += val;
		}

		Console.WriteLine(ans);
	}
}