using Microsoft.ML.Data;

namespace ML.Utilities.Data
{
   /// <summary>
   /// Interfaccia per i providers di opzioni di testo
   /// </summary>
   public interface ITextOptionsProvider
   {
      #region Properties
      /// <summary>
      /// Opzioni di caricamento testi
      /// </summary>
      TextLoader.Options TextOptions { get; }
      #endregion
   }
}
