"""
	ind"..."
	@ind_str "..." "$delim"

Index specification using string syntax.
"""
macro ind_str(s, delim = "")
    [Symbol(i) for i in split(s, delim)]
end
