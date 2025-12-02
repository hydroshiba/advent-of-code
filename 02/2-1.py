from bisect import bisect_left as lower_bound, bisect_right as upper_bound

invalids = []
for num in range(1, 1000000):
	invalids.append(int(str(num) + str(num)))

ranges = input().split(',')
ans = 0

for r in ranges:
	l, r = map(int, r.split('-'))
	
	start = lower_bound(invalids, l, lo = 0, hi = len(invalids))
	end = upper_bound(invalids, r, lo = 0, hi = len(invalids))
	
	value = sum(invalids[start:end])
	ans += value

print(ans)