if len(sorted_coordinates) == 24:
    # for cnt in sorted_coordinates:
    # cv2.rectangle(img, (cnt), (cnt[0] + 1, cnt[1] + 1), (0, 0, 255), 2)

    anchors = TranformPoints(sorted_coordinates, (0, 0), img.shape[0] / 810.0)
    # phan xu li id va sbd
    chieucao = anchors[1][1] - anchors[0][1]
    idtest = [(anchors[0][0], anchors[0][1] - chieucao), (img.shape[0], anchors[1][1])]
    idtest = sub_rect_image(img, idtest)
    list_circle_result = finCircle_idtest(idtest)
    result = get_answers_idtest(list_circle_result)
    print(f'idtest: {result}')
    cv2.putText(img, result, (int(anchors[0][0]), int(anchors[0][1] - chieucao)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1, cv2.LINE_AA)
    # idstudent
    idstudent = [(anchors[2][0], anchors[0][1] - chieucao), anchors[1]]
    idstudent = sub_rect_image(img, idstudent)
    list_circle_result = finCircle_idstudent(idstudent)
    result = get_answers_idstudent(list_circle_result)
    cv2.putText(img, result, (int(anchors[2][0]), int(anchors[0][1] - chieucao)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1, cv2.LINE_AA)
    print(f'idstudent: {result}')

    # phan I
    box_answer = []
    chieurong = anchors[3][0] - anchors[4][0]
    kv1 = [(anchors[5][0] - chieurong, anchors[5][1]), anchors[8]]
    kv2 = [anchors[5], anchors[7]]
    kv3 = [anchors[4], anchors[6]]
    kv4 = [anchors[3], (anchors[6][0] + chieurong, anchors[6][1])]
    img_kv1 = sub_rect_image(img, kv1)
    box_answer.append(img_kv1)
    img_kv2 = sub_rect_image(img, kv2)
    box_answer.append(img_kv2)
    img_kv3 = sub_rect_image(img, kv3)
    box_answer.append(img_kv3)
    img_kv4 = sub_rect_image(img, kv4)
    box_answer.append(img_kv4)
    list_circle = finCircle(box_answer)
    for idx, imgt in enumerate(box_answer):
        cv2.imshow(f'result/20_1_{idx}.png', imgt)

    for idx, imgt in enumerate(list_circle):
        cv2.imwrite(f'result/21_1_{idx}.png', imgt)

    ''' 
    for idx, imgt in enumerate(list_circle):
        cv2.imwrite(f'result/20_1_{idx}.png', imgt)
    phan nay dung de lay ra cac hinh nham muc dich train data
    cau = 0
    for idx, boxtem in enumerate(list_circle):
        cau = cau + 1;
        if (idx + 1) % 4 == 0:
            cau = 0
        ques = idx // 4
        # cv2.imwrite(f'result/a{ques+1}{cau}.png',boxtem)
    '''
    result = get_answers(list_circle)
    print(result)
    # phan II trac nghiem dung sai #
    list_img_ds = []
    ds1 = [(anchors[11][0] - chieurong, anchors[11][1]), anchors[14]]
    ds1 = sub_rect_image(img, ds1)
    list_img_ds.append(ds1)
    ds2 = [anchors[11], anchors[13]]
    ds2 = sub_rect_image(img, ds2)
    list_img_ds.append(ds2)
    ds3 = [anchors[10], anchors[12]]
    ds3 = sub_rect_image(img, ds3)
    list_img_ds.append(ds3)
    ds4 = [anchors[9], (anchors[12][0] + chieurong, anchors[12][1])]
    ds4 = sub_rect_image(img, ds4)
    list_img_ds.append(ds4)
    list_circle_result = []

    for imgitem in list_img_ds:
        list_circle_result.extend(finCircle_ds(imgitem))

    for idx, imgt in enumerate(list_circle_result):
        cv2.imwrite(f'result/21ds_{idx}.png', imgt)

    if len(list_circle_result) == 64:
        result = get_answers_ds(list_circle_result)
        print(result)
    # phan III dien so #
    list_img_tl = []
    chieurong = anchors[15][0] - anchors[16][0]
    tl1 = [(anchors[18][0] - chieurong, anchors[18][1]), anchors[23]]
    tl1 = sub_rect_image(img, tl1)
    cv2.imshow('tl1', tl1)
    tl3 = [anchors[17], anchors[21]]
    tl3 = sub_rect_image(img, tl3)
    tl4 = [(anchors[16][0] - chieurong, anchors[16][1]), anchors[20]]
    tl4 = sub_rect_image(img, tl4)
    tl5 = [anchors[16], anchors[19]]
    tl5 = sub_rect_image(img, tl5)
    tl2 = [anchors[18], anchors[22]]
    tl2 = sub_rect_image(img, tl2)
    cv2.imshow('tl2', tl2)
    tl6 = [anchors[15], (anchors[19][0] + chieurong, anchors[19][1])]
    tl6 = sub_rect_image(img, tl6)
    list_img_tl.append(tl1)
    list_img_tl.append(tl2)
    list_img_tl.append(tl3)
    list_img_tl.append(tl4)
    list_img_tl.append(tl5)
    list_img_tl.append(tl6)
    result_tl = defaultdict(list)

    # for idx, imgt in enumerate(list_img_tl):
    # cv2.imshow(f'{idx}', imgt)
    # list_circle_result = finCircle_tl(imgt)
    # result = get_answers_tl(list_circle_result)
    # result_tl[idx + 1] = result
    cv2.waitKey()
    print(result_tl)
    offset = 0
    for value in result_tl.values():
        cv2.putText(img, value,
                    (int(anchors[18][0] - chieurong + 30 + (offset * chieurong)), int(anchors[18][1] + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
        offset = offset + 1
