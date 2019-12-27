input : segmentation된 도로 전체 이미지, threshold, patch_size

output : 패치 단위 분석 결과 dictionary(full_img_dict) 

=> {patch x좌표, patch y좌표, patch_size_x, patch_size_y, total_max_width, total_average_width, 크랙 minx좌표, 크랙 miny좌표, 크랙 maxx좌표, 크랙 maxy좌표, max_width_x 좌표, max_width_y 좌표, max_width_line x1, max_width_line y1, max_width_line x2, max_width_line y2}

**severity.py**

균열의 심각도를 구하는 작업의 코드

#

input : 패치 이미지, x좌표, y좌표

output : 빨간색 직사각형이 좌표에 그려진 패치 이미지

**draw_rectangle 함수**

패치 이미지와 x,y좌표를 받아서 해당 좌표에 빨간 색 직사각형을 그려 반환하는 함수

#

input : A(패치 이미지), xCenter, yCenter, radius

output : circle_x_y_color(원의 점 좌표를의 리스트)

**get_circle_x_y_color_list 함수**

중심 x,y좌표를 기준으로 반지름 입력을 크기로하는 점들의 좌표를 반환하는 함수(get_is_joint 함수의 내부 함수)

#

input : A,xCenter,yCenter,radius

output : A,is_joint,reverse_count

**get_is_joint 함수**

패치 이미지에서 해당 좌표가 joint(균열 접합지점)인지 검출하는 함수 반환값은 A(패치 이미지), is_joint(조인트 여부), reverse_count(균열과 배경의 경계 지점 검출 횟수= 5이상일 경우 조인트 지점으로 판별)

#

input : A,max_width_dict

output : x, y, value

**get_max_width_x_max_width_y 함수**

패치 이미지와 max_width 딕셔너리를 입력으로 받아 조인트가 아닌 최대 좌표 값(x,y,value)를 출력하는 함수

#

input : arr,base_x,base_y,amount_of_change_x,amount_of_change_y,size_x,size_y

output : count

**count_length 함수**

crack이 존재하는 픽셀의 인덱스 좌표에 전체 적용되는 함수(base좌표를 기준으로 x,y방향으로 x,y증가량을 적용시켜 crack의 폭을 계산하는 함수)

#

input : A

output : A,avg_crack_width,minx,miny,maxx,maxy,max_width_dict

**every_search_get_max_avg_width_of_crack 함수**

severity.py에서 crack이 존재하는 픽셀의 인덱스 좌표를 따라 4방향(상하좌우)의 균열이 존재하는 픽셀 길이를 구하고 최소값을 crack_width에 append해서 저장하는 함수

#

input : A,base_x,base_y,amount_of_change_x,amount_of_change_y,size_x,size_y,color

output : A

**visualize_extended_line 함수**

count_length함수가 어떻게 그려지는 시각화하기 위한 함수

#

input : A,base_x,base_y,amount_of_change_x,amount_of_change_y,size_x,size_y

output : pick_x,pick_y

**get_extended_line_x_y 함수**

기준이 되는 x,y 좌표에서 균열 폭 시작지점 좌표와 끝지점 좌표를 출력하기 위한 함수 

#

input : A,output_path,fname

output : None

**save 함수**

패치 이미지를 저장하는 함수

#

input : A,thres

output : A

**binaryize 함수**

패치 이미지를 threshold 값으로 이진화하는 함수

#

input : A,x,y,width,height

output : 상수(0,1,2,3중의 하나)

**find_min_direction 함수**

4방향으로 count_length함수를 적용해 크기가 작은 방향을 검출하는 함수

#

input : A,x_y_color

output : A

**visualize_circle 함수**

원을 그리는 함수

#

input : A,count,return_list,patch_size,i_j

output : return_list[count] = {
                'x':patch_size*i,
                'y':patch_size*j,
                'w':patch_size,
                'h':patch_size,
                'total_max_width':total_max_width,
                'total_average_width':total_average_width,
                'minx':minx,
                'miny':miny,
                'maxx':maxx,
                'maxy':maxy,
                'max_width_x':max_width_x,
                'max_width_y':max_width_y,
                'line_x1':line_x1,
                'line_y1':line_y1,
                'line_x2':line_x2,
                'line_y2':line_y2
            }

**full_process 함수**

severity.py에서 main이 되는 함수(분석결과를 dictionary로 반환)
#
