from bisect import bisect_left as lower_bound, bisect_right as upper_bound

invalids = []
for num in range(1, 1000000):
	string = str(num)
	for _ in range(12):
		string += str(num)
		if(len(string) > 12): break
		invalids.append(int(string))

invalids = sorted(list(set(invalids)))

ranges = input().split(',')
ans = 0

for r in ranges:
	l, r = map(int, r.split('-'))
	
	start = lower_bound(invalids, l, lo = 0, hi = len(invalids))
	end = upper_bound(invalids, r, lo = 0, hi = len(invalids))
	
	value = sum(invalids[start:end])
	ans += value

print(ans)