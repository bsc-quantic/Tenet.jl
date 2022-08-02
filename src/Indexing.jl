"""
	cind"..."

`Char`-based index specification.
"""
macro cind_str(s)
    [Symbol(i) for i in s]
end
