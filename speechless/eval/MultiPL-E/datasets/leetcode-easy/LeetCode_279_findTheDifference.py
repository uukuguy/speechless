def findTheDifference(s: str, t: str) -> str:
    """
    You are given two strings s and t.
    String t is generated by random shuffling string s and then add one more letter at a random position.
    Return the letter that was added to t.
 
    Example 1:

    Input: s = "abcd", t = "abcde"
    Output: "e"
    Explanation: 'e' is the letter that was added.

    Example 2:

    Input: s = "", t = "y"
    Output: "y"

 
    Constraints:

    0 <= s.length <= 1000
    t.length == s.length + 1
    s and t consist of lowercase English letters.

    """
    ### Canonical solution below ###
    return chr(sum(ord(c) for c in t) - sum(ord(c) for c in s))




### Unit tests below ###
def check(candidate):
	assert candidate("swift", "switft") == "t"
	assert candidate("helloworld", "hellowordll") == "l"
	assert candidate("a", "aa") == "a"
	assert candidate("helloworld", "helloworldl") == "l"
	assert candidate("css", "scsx") == "x"
	assert candidate(
    "html", "htmln") == "n"
	assert candidate("abcd", "abcde") == "e"
	assert candidate("nosql", "onsqln") == "n"
	assert candidate("css", "cssc") == "c"
	assert candidate("docker", "dockere") == "e"
	assert candidate("sql", "sqlq") == "q"
	assert candidate("python", "ypthnoa") == "a"
	assert candidate("xml", "lxmm") == "m"
	assert candidate("json", "osnjn") == "n"
	assert candidate(
    "python", "pythonn") == "n"
	assert candidate("a", "ab") == "b"
	assert candidate("html", "thmla") == "a"
	assert candidate("abc", "bacd") == "d"
	assert candidate("", "b") == "b"
	assert candidate("l", "lw") == "w"
	assert candidate("abc", "abcd") == "d"
	assert candidate("hello", "ohelll") == "l"
	assert candidate("aab", "aabb") == "b"
	assert candidate("", "y") == "y"
	assert candidate("", "q") == "q"
def test_check():
	check(findTheDifference)
# Metadata Difficulty: Easy
# Metadata Topics: hash-table,string,bit-manipulation,sorting
# Metadata Coverage: 100
