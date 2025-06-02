# cin
cin使用空白（空格，制表符和换行符）来确定字符串的结束位置，这意味着cin会自动在字符串结尾添加空字符。
所以说如果我们希望能够完整读入一行数据，比如说new york，那么此时我们可以使用cin.getline()以及cin.get(),它们都提取一行输入，直到换行符，getline最终会丢弃换行符，而get会将换行符保存在输入队列中。                    
假设使用getline将姓名读入到一个包含20个元素的name数组中，可以写为：cin.getline(name, 20)；cin.get在使用时会存在有一个换行没有舍弃的情况，我们可以再写一个cin.get()吃掉它，即cin.get(name, 20).get()；也可以string完之后直接写getline（cin， str）这样字符串不受字数的限制。
cout<<R"(jim "king"uses"\n")"会直接全部打印出来含有“与\n。
