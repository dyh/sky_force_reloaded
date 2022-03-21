import numpy as np
import win32gui, win32con
import win32ui
import cv2

import config
from sky_force_reloaded.object_detection import ObjectDetection


def run_battle_observation():
    """
    开始
    :return:
    """
    print('battle observation process starting...')

    # 获取后台窗口的句柄，注意后台窗口不能最小化
    # 窗口的类名可以用Visual Studio的SPY++工具获取
    h_wnd = win32gui.FindWindow(config.game_window_class_name, config.game_window_title)

    # 获取句柄窗口的大小信息
    left, top, right, bottom = win32gui.GetWindowRect(h_wnd)
    width = right - left
    height = bottom - top

    # 返回句柄窗口的设备环境，覆盖整个窗口，包括非客户区，标题栏，菜单，边框
    h_wnd_dc = win32gui.GetWindowDC(h_wnd)
    # 创建设备描述表
    mfc_dc = win32ui.CreateDCFromHandle(h_wnd_dc)
    # 创建内存设备描述表
    save_dc = mfc_dc.CreateCompatibleDC()
    # 创建位图对象准备保存图片
    save_bitmap = win32ui.CreateBitmap()
    # 为bitmap开辟存储空间
    save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)

    # 循环截图
    # self._loop_screen(save_dc, save_bitmap, width, height, mfc_dc, h_wnd)

    # 实例化ObjectDetection
    obj_detect = ObjectDetection()

    # # 实例化BattleThinking
    # battle_think = BattleThinking()
    #
    # # 实例化HeroMoveProcess
    # hero_move_process = HeroMoveProcess(h_wnd)
    # hero_move_process.start_process()
    #
    # # 实例化HeroAttackProcess
    # hero_attack_process = HeroAttackProcess(h_wnd)
    # hero_attack_process.start_process()

    # # 读取 游戏按钮等 模板，用于匹配
    # template_button_continue = cv2.imread(r".\brawl_stars\images\continue.jpg", flags=0)
    # template_button_exit = cv2.imread(r".\brawl_stars\images\exit.jpg", flags=0)
    # template_button_fight = cv2.imread(r".\brawl_stars\images\fight.jpg", flags=0)

    # 判断 进程间共享变量 是否为 True
    while config.LOOP_ACTIVE_FLAG:
        # 将截图保存到saveBitMap中
        save_dc.SelectObject(save_bitmap)
        # 保存bitmap到内存设备描述表
        save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

        signed_ints_array = save_bitmap.GetBitmapBits(True)

        # img = np.fromstring(signed_ints_array, dtype='uint8')

        img = np.frombuffer(signed_ints_array, dtype='uint8')

        img.shape = (height, width, 4)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # # [battle_observation]进程，调用object_detection功能，对图片进行检测，返回物体信息list，
        # 如果显示预览窗体，则 画框
        list_detect, img = obj_detect.detect(img, draw_box=config.display_preview_screen)

        if len(list_detect) > 0:
            print(list_detect)
        pass

        # 是否显示游戏预览
        if config.display_preview_screen:
            # 全屏显示游戏画面
            if config.preview_full_screen:
                cv2.namedWindow(config.cv2_window_title, flags=cv2.WND_PROP_FULLSCREEN)
                # cv2.moveWindow(config.cv2_window_title, screen.x - 1, screen.y - 1)
                cv2.setWindowProperty(config.cv2_window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            pass

            # 在指定窗口中，显示图片
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

            cv2.imshow(config.cv2_window_title, img)
            # 0 表示程序会无限制的等待用户的按键事件
            key = cv2.waitKey(1)

            if key != -1:
                print(key)

                if key == 113:
                    # q 键 退出
                    config.LOOP_ACTIVE_FLAG = False
                    pass
                pass
            pass
        else:
            pass
        pass

        # # 如果检测到物体，则移动或战斗
        # if len(list_detect) > 0:
        #     # [battle_observation]进程，调用battle_thinking功能，对物体信息list进行分析
        #     # 清空数据
        #     battle_think.clear_data()
        #     # 处理物体列表
        #     result_move_direction, result_move_distance, result_attack_direction, result_attack_type = \
        #         battle_think.process_all(objects_list=list_detect)
        #
        #     # [battle_observation]进程，启动[hero_movement]进程，由[hero_movement]进程，调用[device_control]功能，实现移动功能。
        #     hero_move_process.refresh(move_direction=result_move_direction, move_distance=result_move_distance)
        #
        #     # [battle_observation]进程，启动[hero_attack]进程，由[hero_attack]进程，调用[device_control]功能，实现攻击功能。
        #     hero_attack_process.refresh(attack_direction=result_attack_direction, attack_type=result_attack_type)
        # else:
        #     # 如果没有检测到物体，则判断 是否 进入了游戏菜单的控制流程
        #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        #     # game_control_flag = self._is_contained(img, )
        #     # template_button_continue = cv2.imread(r".\brawl_stars\images\continue.jpg", flags=0)
        #     # template_button_exit = cv2.imread(r".\brawl_stars\images\exit.jpg", flags=0)
        #     # template_button_fight = cv2.imread(r".\brawl_stars\images\fight.jpg", flags=0)
        #     # 图片比对的阈值
        #     value = 0.7
        #
        #     # 继续 按钮
        #     position_continue_button = self._get_button_position(img_gray, template_button_continue, value)
        #     if position_continue_button is not None:
        #         # 点击按钮
        #         x, y, w, h = position_continue_button
        #         # 这里用 预设在模拟器中的 虚拟键盘 点击 继续 按钮，也可以换成鼠标点击
        #         menu_control.click_continue_button(h_wnd)
        #
        #         # 进行下一次循环
        #         continue
        #
        #     # 退出 按钮
        #     position_exit_button = self._get_button_position(img_gray, template_button_exit, value)
        #     if position_exit_button is not None:
        #         # 点击按钮
        #         x, y, w, h = position_exit_button
        #         # 这里用 预设在模拟器中的 虚拟键盘 点击 继续 按钮，也可以换成鼠标点击
        #         menu_control.click_exit_button(h_wnd)
        #
        #         # 进行下一次循环
        #         continue
        #
        #     # 对战 按钮
        #     position_fight_button = self._get_button_position(img_gray, template_button_fight, value)
        #     if position_fight_button is not None:
        #         # 点击按钮
        #         x1, y1, w1, h1 = position_fight_button
        #         # 这里用 预设在模拟器中的 虚拟键盘 点击 继续 按钮，也可以换成鼠标点击
        #         menu_control.click_fight_button(h_wnd)
        #
        #         # 进行下一次循环
        #         continue
        #     pass
    pass

    # 释放cv2
    cv2.destroyWindow(config.cv2_window_title)

    # 释放资源
    mfc_dc.DeleteDC()
    save_dc.DeleteDC()
    win32gui.ReleaseDC(h_wnd, h_wnd_dc)
    win32gui.DeleteObject(save_bitmap.GetHandle())

    print('battle observation process terminated.')
    pass


if __name__ == '__main__':
    config.LOOP_ACTIVE_FLAG = True
    run_battle_observation()
    print('abc')
