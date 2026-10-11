from arabic_voxcpm.text import clean, strip_dialect_tag


def test_clean():
    assert strip_dialect_tag("[EG] إزيك يا صاحبي") == "إزيك يا صاحبي"
    assert clean("[BH] شراکات فی القطاع") == "شراكات في القطاع"
    assert clean("[SA] من صومعته sigh") == "من صومعته"
    assert clean("كـــبير  [exhale]  جدا") == "كبير جدا"
    assert clean("[EG] باستخدام الـ AI في الشغل") == "باستخدام ال AI في الشغل"
