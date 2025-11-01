#!/usr/bin/env python
import sys
try:
    from tree_sitter import Parser
    from tree_sitter_languages import get_language
    
    # 测试代码
    code = b"function test() { return 42; }"
    
    # 获取语言和解析器
    lang = get_language("javascript")
    parser = Parser()
    parser.set_language(lang)
    
    # 解析
    tree = parser.parse(code)
    
    # 测试query
    query = lang.query("(function_declaration) @func")
    captures = query.captures(tree.root_node)
    
    print(f"✓ tree-sitter working!")
    print(f"  Parsed: {len(captures)} function(s)")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
