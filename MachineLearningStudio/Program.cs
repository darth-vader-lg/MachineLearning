using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace TestChoice
{
   /// <summary>
   /// Programma
   /// </summary>
   internal static class Program
   {
      #region Properties
      /// <summary>
      /// Impostazioni
      /// </summary>
      public static Settings Settings { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Punto di ingresso principale dell'applicazione.
      /// </summary>
      [STAThread]
      private static void Main()
      {
         Application.EnableVisualStyles();
         Application.SetCompatibleTextRenderingDefault(false);
         Settings = new Settings();
         try {
            Settings.Load(Path.Combine(Environment.CurrentDirectory, "Settings.xml"));
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
         Application.Run(new MainForm());
      }
      #endregion
   }
}
