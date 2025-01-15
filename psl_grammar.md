
# PSL grammar

<program>               ::= <declarations> <statements>
<declarations>          ::= <declaration> [ <declarations> ]
<declaration>           ::= <register map> | <constant map> | <system map> | <watch list>

<register map>          ::= registers "{" <register entry list> "}"
<register entry list>   ::= <register entry> [ "," <register entry list> ]
<register entry>        ::= <register name> ":" <register short name>
<register name>         ::= <unquoted string>
<register short name>   ::= <quoted string>

<constant map>          ::= constants "{" <constant entry list> "}"
<constant entry list>   ::= <constant entry> [ "," <constant entry list> ]
<constant entry>        ::= <constant name> [ ":" <quoted string> ]
<constant name>         ::= <unquoted string>

<system map>            ::= system "{" <system entry list> "}"
<system entry list>     ::= <system entry> [ "," <system entry list> ]
<system entry>          ::= <system register name> ":" <register name>
<system register name>  ::= symbol | position | output | parse | eop

<watch list>            ::= watch "[" <register name> [ "," <register name> ] "]"

<statements>            ::= <statement> [ <statements> ]
<statement>             ::=  <causal attn statement> | <where statement> | <repeat statement>

<causal attn statement> ::= causal_attn ":" <boolean value>
<boolean value>         ::= true | false

<where statement>       ::= <where variant> <conditions> ":" <assignments>
<where variants>        ::= where | where_lm | where_rm
            
<conditions>            ::= <condition> [ and <conditions> ]
<condition>             ::= <simple condition> | "(" <conditions> ")"    

<simple condition>      ::= <bool compare> | <bool in>
<bool compare>          ::= <left cond> <compare op> <right cond>   
<left cond>             ::= <register name> "[" <register index> "]"
<register index>        ::= N | n 
<compare op>            ::= "==" | "!="
<right cond>            ::= <constant name> | <right reg> [ <weight func> ]
<right reg>             ::= <register name> "[" <register index> "]" 
<weight func>           ::= "@" <weight function> 
<weight function>       ::= pos_increment | pos_decrement

<bool in>               ::= <left cond> <in op> "[" <constant list> "]"
<in op>                 ::= in | not in
<constant list>         ::= <constant name> [ "," <constant list> ]

<assignment list>       ::= <assignment> [ <assignment list> ]
<assignment>            ::= <assign left> "=" <assign right>
<assign left>           ::= <register name> "[" N "]" 
<assign right>          ::= <constant name> | <right reg>

<repeat statement>      ::= repeat <statements> until <stop condition>
<stop condition>        ::= NO_CHANGE | <conditions>