"""Optional real-browser check: run against a server on port 8060.

Requires Playwright and installed Microsoft Edge. Run with python tests/browser_check.py.
This stays separate from pytest so CI does not need a browser download.
"""
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge',headless=True)
        page = browser.new_page(viewport={'width':1440,'height':1100})
        errors = []
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:8060/')
        page.locator('#auth-guest-btn').click()
        expect(page.locator('.square')).to_have_count(64)
        artifacts = Path('artifacts')
        artifacts.mkdir(exist_ok=True)
        page.screenshot(path=str(artifacts/'chess-desktop.png'),full_page=True)
        page.get_by_text('Two players',exact=True).click()
        expect(page.locator('#opponent')).to_contain_text('Local player')

        def wait_move(count):
            expect(page.locator('#move-count')).to_have_text(f'{count} move' + ('' if count == 1 else 's'))
            expect(page.locator('#board')).to_have_attribute('data-busy','false')

        def drag(source,target):
            start = page.locator('#square-'+source).bounding_box()
            end = page.locator('#square-'+target).bounding_box()
            page.mouse.move(start['x']+start['width']/2,start['y']+start['height']/2)
            page.mouse.down()
            page.mouse.move(end['x']+end['width']/2,end['y']+end['height']/2,steps=12)
            page.mouse.up()

        page.locator('#square-e2').click()
        expect(page.locator('#square-e4')).to_have_class('square light legal-target')
        page.locator('#square-e4').click()
        wait_move(1)
        drag('g8','f6')
        wait_move(2)
        drag('e4','e6')
        expect(page.locator('#feedback')).to_contain_text('not available')
        wait_move(2)
        drag('e4','e5')
        wait_move(3)
        start = page.locator('#square-g7').bounding_box()
        page.mouse.move(start['x']+start['width']/2,start['y']+start['height']/2)
        page.mouse.down()
        page.mouse.move(10,10,steps=10)
        page.mouse.up()
        expect(page.locator('.drag-ghost')).to_have_count(0)
        wait_move(3)
        page.locator('#reset').click()
        wait_move(0)
        page.locator('#square-e2').focus()
        page.keyboard.press('Enter')
        expect(page.locator('#square-e2')).to_have_attribute('aria-pressed','true')
        page.locator('#square-e4').focus()
        page.keyboard.press('Enter')
        wait_move(1)
        page.reload()
        wait_move(1)
        expect(page.locator('#opponent')).to_contain_text('Local player')
        page.locator('#reset').click()
        wait_move(0)
        page.locator('summary').click()
        for count,san in enumerate(['a4','h5','a5','h4','a6','h3','axb7','hxg2'],1):
            page.locator('#notation').fill(san)
            page.locator('#play').click()
            wait_move(count)
        page.locator('#promotion').click()
        page.get_by_text('Knight',exact=True).click()
        drag('b7','a8')
        wait_move(9)
        expect(page.locator('#square-a8')).to_have_attribute('aria-label','a8, white knight')
        page.screenshot(path=str(artifacts/'chess-midgame.png'),full_page=True)
        page.get_by_text('Vs computer',exact=True).click()
        wait_move(0)
        drag('d2','d4')
        wait_move(2)
        expect(page.locator('#status')).to_have_text('White to move.')

        # A touch-capable viewport uses the same pointer handlers.
        touch = browser.new_context(viewport={'width':390,'height':844},is_mobile=True,has_touch=True)
        mobile = touch.new_page()
        mobile.goto('http://127.0.0.1:8060/')
        mobile.locator('#auth-guest-btn').tap()
        expect(mobile.locator('.square')).to_have_count(64)
        mobile.get_by_text('Two players',exact=True).tap()
        expect(mobile.locator('#opponent')).to_contain_text('Local player')
        rects = [mobile.locator('#square-'+s).bounding_box() for s in ('e2','e4')]
        client = touch.new_cdp_session(mobile)
        def point(rect):
            return {'x':rect['x']+rect['width']/2,'y':rect['y']+rect['height']/2}
        start,end = map(point,rects)
        client.send('Input.dispatchTouchEvent',{'type':'touchStart','touchPoints':[start]})
        client.send('Input.dispatchTouchEvent',{'type':'touchMove','touchPoints':[end]})
        client.send('Input.dispatchTouchEvent',{'type':'touchEnd','touchPoints':[]})
        expect(mobile.locator('#move-count')).to_have_text('1 move')
        assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        mobile.screenshot(path=str(artifacts/'chess-mobile.png'),full_page=True)
        assert not errors, errors
        browser.close()
        print('Browser checks passed: click, drag, illegal drop, keyboard, session replay, underpromotion and touch.')


if __name__ == '__main__':
    main()
