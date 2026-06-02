import asyncio
import browser_cookie3
from playwright.async_api import async_playwright

async def main():
    # 1. 获取 Edge 的 AI Studio cookies
    cj = browser_cookie3.edge(domain_name='aistudio.baidu.com')
    cookies = list(cj)
    print(f"获取 {len(cookies)} 个 cookies")

    # 2. 启动 Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        # 3. 添加 cookies
        pw_cookies = []
        for c in cookies:
            pw_cookies.append({
                "name": c.name,
                "value": c.value,
                "domain": c.domain,
                "path": c.path if c.path else "/",
                "secure": bool(c.secure),
            })
        await context.add_cookies(pw_cookies)

        page = await context.new_page()

        # 4. 访问 AI Studio 项目页面
        print("正在访问 AI Studio...")
        await page.goto("https://aistudio.baidu.com/projectdetail/10186630", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(5000)

        title = await page.title()
        print(f"页面标题: {title}")

        # 5. 检查登录状态
        body_text = await page.inner_text("body")
        if "松柏ip" in body_text:
            print("状态: 已登录 ✓")
        elif "登录" in body_text[:500]:
            print("状态: 未登录 ✗")
        else:
            print(f"状态: 未知 (前200字: {body_text[:200]})")

        # 6. 截图
        await page.screenshot(path="/Users/sunyiyang/Desktop/Project/Baidu  GRAB/aistudio_page.png", full_page=False)
        print("截图已保存")

        await browser.close()

asyncio.run(main())
