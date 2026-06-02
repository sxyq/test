import asyncio
from playwright.async_api import async_playwright

EDGE_USER_DATA = "/Users/sunyiyang/Library/Application Support/Microsoft Edge"
AI_STUDIO_PROJECT = "https://aistudio.baidu.com/projectdetail/10186630"

async def main():
    async with async_playwright() as p:
        # 使用 Edge 的用户数据目录启动，保留登录状态
        context = await p.chromium.launch_persistent_context(
            user_data_dir=EDGE_USER_DATA,
            executable_path="/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            headless=False,
            channel="msedge",
            viewport={"width": 1280, "height": 800},
            args=["--disable-blink-features=AutomationControlled"],
        )

        page = context.pages[0] if context.pages else await context.new_page()

        # 访问 AI Studio 项目页面
        print("正在访问 AI Studio 项目页面...")
        await page.goto(AI_STUDIO_PROJECT, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)

        # 检查登录状态
        title = await page.title()
        print(f"页面标题: {title}")

        body_text = await page.inner_text("body")
        if "登录" in body_text[:800] and "松柏" not in body_text:
            print("状态: 未登录 ✗")
            # 截图保存
            await page.screenshot(path="/Users/sunyiyang/Desktop/Project/Baidu  GRAB/aistudio_login.png")
            print("截图已保存到 aistudio_login.png")
        else:
            print("状态: 已登录 ✓")
            # 截图保存
            await page.screenshot(path="/Users/sunyiyang/Desktop/Project/Baidu  GRAB/aistudio_logged_in.png")
            print("截图已保存到 aistudio_logged_in.png")

            # 查找 Fork 按钮
            print("\n查找页面元素...")
            try:
                fork_btn = page.locator("text=Fork").first
                if await fork_btn.is_visible(timeout=5000):
                    print("找到 Fork 按钮")
                    await fork_btn.click()
                    print("已点击 Fork 按钮")
                    await page.wait_for_timeout(10000)
                    await page.screenshot(path="/Users/sunyiyang/Desktop/Project/Baidu  GRAB/aistudio_after_fork.png")
                    print("Fork 后截图已保存")
                else:
                    print("Fork 按钮不可见")
            except Exception as e:
                print(f"Fork 操作异常: {e}")

            # 列出页面上可交互的元素
            print("\n页面主要按钮/链接:")
            buttons = await page.query_selector_all("button, a[role='button'], .ant-btn")
            for btn in buttons[:20]:
                text = await btn.inner_text()
                if text.strip():
                    print(f"  - {text.strip()[:50]}")

        # 保持浏览器打开 30 秒供查看
        print("\n浏览器将保持打开 30 秒，你可以手动操作...")
        await page.wait_for_timeout(30000)

        await context.close()
        print("浏览器已关闭")

asyncio.run(main())
