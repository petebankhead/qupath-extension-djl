/*-
 * Copyright 2023 QuPath developers, University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package qupath.ext.djl.ui;

import ai.djl.util.Utils;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.StringProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.fx.prefs.annotations.BooleanPref;
import qupath.fx.prefs.annotations.DirectoryPref;
import qupath.fx.prefs.annotations.Pref;
import qupath.fx.prefs.annotations.PrefCategory;
import qupath.fx.prefs.annotations.StringPref;
import qupath.lib.gui.prefs.PathPrefs;

import java.util.Arrays;
import java.util.List;

@PrefCategory(bundle="qupath.ext.djl.ui.strings", value="prefs.category")
public class DjlPrefs {

    private static final Logger logger = LoggerFactory.getLogger(DjlPrefs.class);

    /**
     * System property to specify the PyTorch version to use.
     */
    public static final String KEY_PYTORCH_VERSION = "PYTORCH_VERSION";

    /**
     * System property to specify the PyTorch library path.
     */
    private static final String KEY_PYTORCH_LIBRARY_PATH = "PYTORCH_LIBRARY_PATH";

    /**
     * System property to specify the PyTorch flavor (e.g. cpu, cu118)
     */
    private static final String KEY_PYTORCH_FLAVOR = "PYTORCH_FLAVOR";

    /**
     * System property to request PyTorch to use pre-C++11 ABI (should be "true" or "false").
     */
    private static final String KEY_PYTORCH_PRECXX11 = "PYTORCH_PRECXX11";

    @BooleanPref(bundle="qupath.ext.djl.ui.strings", value="prefs.offline")
    private final BooleanProperty createOfflineProperty = createSystemBooleanProperty("djl.offline", "offline", true);

    @StringPref(bundle="qupath.ext.djl.ui.strings", value="prefs.jna.path")
    private final StringProperty jnaPath = createSystemPropertyPref("djl.jnaPath", "jna.library.path");

    @Pref(bundle="qupath.ext.djl.ui.strings", type=String.class, value="prefs.pytorch.version", choiceMethod = "getPyTorchVersions")
    private final StringProperty pytorchVersion = createSystemPropertyPref("djl.pytorchVersion", KEY_PYTORCH_VERSION);

    @DirectoryPref(bundle="qupath.ext.djl.ui.strings", value="prefs.pytorch.path")
    private final StringProperty pytorchPath = createSystemPropertyPref("djl.pytorchLibraryPath", KEY_PYTORCH_LIBRARY_PATH);

    @StringPref(bundle="qupath.ext.djl.ui.strings", value="prefs.pytorch.flavor")
    private final StringProperty pytorchFlavor = createSystemPropertyPref("djl.pytorchFlavor", KEY_PYTORCH_FLAVOR);

    @BooleanPref(bundle="qupath.ext.djl.ui.strings", value="prefs.pytorch.precxx11")
    private final BooleanProperty pytorchPreCxx11 = createSystemBooleanProperty("djl.pytorchPreCxx11", KEY_PYTORCH_PRECXX11, false);

    /**
     * Get the available versions of PyTorch supported by Deep Java Library.
     * @return
     */
    public List<String> getPyTorchVersions() {
        return Arrays.asList("", "1.12.1", "1.13.1", "2.0.1");
    }

    private static DjlPrefs instance = new DjlPrefs();

    private DjlPrefs() {}

    private static StringProperty createSystemPropertyPref(String name, String key) {
        var val = Utils.getEnvOrSystemProperty(key);
        if (val != null && val.equals(System.getenv(key)))
            logger.info("{}={} set through environment variable, changes will be ignored", key, val);
        var prop = PathPrefs.createPersistentPreference(name, val);
        prop.addListener((v, o, n) -> updateSystemProperty(key, n));
        updateSystemProperty(key, prop.getValue()); // May be different because of stored preference
        return prop;
    }

    private static void updateSystemProperty(String key, String value) {
        if (value == null || value.isEmpty())
            System.clearProperty(key);
        else
            System.setProperty(key, value);
    }

    private static BooleanProperty createOfflineProperty(String name, String key) {
        var val = Utils.getEnvOrSystemProperty(key);
        // Default should be true, unless we have a value set somewhere else
        boolean defaultValue = !"false".equalsIgnoreCase(val);
        var prop = PathPrefs.createPersistentPreference(name, defaultValue);
        prop.addListener((v, o, n) -> updateSystemProperty(key, n ? "true" : "false"));
        updateSystemProperty(key, prop.getValue() ? "true" : "false"); // May be different because of stored preference
        return prop;
    }

    private static BooleanProperty createSystemBooleanProperty(String name, String key, boolean defaultValue) {
        var val = Utils.getEnvOrSystemProperty(key);
        // Default should be true, unless we have a value set somewhere else
        if (val != null && !val.isEmpty())
            defaultValue = "true".equalsIgnoreCase(val);
        var prop = PathPrefs.createPersistentPreference(name, defaultValue);
        prop.addListener((v, o, n) -> updateSystemProperty(key, n ? "true" : "false"));
        updateSystemProperty(key, prop.getValue() ? "true" : "false"); // May be different because of stored preference
        return prop;
    }

    public static DjlPrefs getInstance() {
        return instance;
    }

}
