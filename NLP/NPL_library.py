def check_keyword (original_keyword, keyword):
    # print('--in 1--')
    # print(original_keyword, keyword)

    lst = list(keyword)
    input_keyword = original_keyword.split('_')
    for word_divide in input_keyword:
        if keyword[0] in word_divide:
            lst[0] = word_divide

    keyword = tuple(lst)

    npl_word = ""
    # print('--in 2--')
    # print(original_keyword, keyword[0])
    if keyword[0] == '제주':
        # print('--in 3--')
        npl_word = '제주도'
    else:
        npl_word = keyword[0]

    if keyword[1] not in ('N','F'):
        npl_word = ""

    # print('--in 4--')
    # print(npl_word)
    return npl_word