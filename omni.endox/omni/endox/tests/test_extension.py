import omni.kit.test


class TestEndoxPipeline(omni.kit.test.AsyncTestCaseFailOnLogError):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_extension_loads(self):
        import omni.endox
        self.assertIsNotNone(omni.endox)
