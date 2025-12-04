program AdventOfCode;
var n, m: longint; grid: array[1..1000] of widestring; // 1-indexed bruhhhhhhh

function satisfy(const x, y: longint): boolean;
var cnt: longint;
begin
	cnt := 0;

	if (x > 1) and (y > 1) and (grid[x - 1, y - 1] = '@') then inc(cnt);
	if (x > 1) and (grid[x - 1, y] = '@') then inc(cnt);
	if (x > 1) and (y < m) and (grid[x - 1, y + 1] = '@') then inc(cnt);
	if (y > 1) and (grid[x, y - 1] = '@') then inc(cnt);
	if (y < m) and (grid[x, y + 1] = '@') then inc(cnt);
	if (x < n) and (y > 1) and (grid[x + 1, y - 1] = '@') then inc(cnt);
	if (x < n) and (grid[x + 1, y] = '@') then inc(cnt);
	if (x < n) and (y < m) and (grid[x + 1, y + 1] = '@') then inc(cnt);

	satisfy := (cnt < 4) and (grid[x, y] = '@');
end;

procedure solve();
var i, j: integer; ans: longint;
begin
	n := 1;
	ans := 0;

	while not eof() do
	begin
		readln(grid[n]);
		m := length(grid[n]);
		inc(n);
	end;

	dec(n);

	for i := 1 to n do
		for j := 1 to m do
			if satisfy(i, j) then
				inc(ans);

	writeln(ans);
end;

begin
	solve();
end.