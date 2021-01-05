using Microsoft.ML;

namespace MachineLearning
{
   /// <summary>
   /// Interfaccia per il filtro di righe di una vista di dati
   /// </summary>
   public interface IDataViewRowFilter
   {
      #region Methods
      /// <summary>
      /// Restituisce lo stato di validita' di una riga  di dati
      /// </summary>
      /// <param name="row">Riga di dati</param>
      /// <returns>true se la riga e' valida</returns>
      bool IsValidRow(DataViewRow row);
      #endregion
   }

   /// <summary>
   /// Filtro di righe di dati
   /// </summary>
   /// <param name="row">Riga</param>
   /// <returns>true se la riga e' valida</returns>
   public delegate bool DataViewRowFilter(DataViewRow row);
}
