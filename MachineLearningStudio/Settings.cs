using System;
using System.Diagnostics;
using System.IO;

namespace TestChoice
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
            Default.Load(Path.Combine(Environment.CurrentDirectory, "Settings.xml"));
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      #endregion
   }
}
