using System;
using System.Diagnostics;
using System.IO;

namespace MachineLearningStudio
{
   /// <summary>
   /// Impostazioni applicazione
   /// </summary>
   [Serializable]
   public partial class Settings : XmlSettings
   {
      #region Properties
      /// <summary>
      /// Istanza globale di default
      /// </summary>
      public static Settings Default { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore statico
      /// </summary>
      static Settings()
      {
         Default = new Settings();
         try {
            var path = Path.Combine(Environment.CurrentDirectory, "Settings.xml");
            if (!File.Exists(path))
               Default.Save(path);
            else
               Default.Load(path);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      #endregion
   }
}
